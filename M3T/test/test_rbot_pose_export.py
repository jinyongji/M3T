"""Run with -B; set RBOT_POSE_TEST_DIR for retained fixtures in the result root."""
import csv
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pose_analysis", PROJECT / "analyze_rbot_poses.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PoseAnalysisTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ["RBOT_POSE_TEST_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="pose_fixture_", dir=root))
        self.path = self.directory / "poses_c_noisy_test.csv"
        self.metadata = {
            "schema_version": 1, "translation_unit": "m", "object": "test",
            "sequence": "c_noisy", "frame_count": 2, "first_image_index": 1,
            "last_image_index": 2, "pose_convention": "p_camera = R * p_body + t",
            "reset_protocol": "rbot_selected_output_reset",
        }
        identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        quarter_turn = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        self.rows = []
        for frame in range(2):
            row = {
                "frame": frame, "image_index": frame + 1,
                "image_file": f"c_noisy{frame+1:04d}.png", "segment_id": frame,
                "estimate_source": "single", "pose_finite": 1,
                "tracking_success": frame, "reset_primary_after_frame": 1 - frame,
                "reset_secondary_after_frame": 0,
                "translation_error_m": 0.1 if frame == 0 else 0,
                "rotation_error_rad": math.pi / 2 if frame == 0 else 0,
            }
            for prefix in ("est", "gt"):
                rotation = quarter_turn if prefix == "est" and frame == 0 else identity
                for r in range(3):
                    for c in range(3):
                        row[f"{prefix}_r{r}{c}"] = rotation[3 * r + c]
                for axis in "xyz":
                    row[f"{prefix}_t{axis}_m"] = 0.1 if prefix == "est" and frame == 0 and axis == "x" else 0
            self.rows.append(row)

    def save(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        (self.directory / "metadata_c_noisy_test.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_metrics_and_custom_threshold(self):
        self.save()
        result = analysis.analyze_file(self.path, 0.05, 5)
        self.assertAlmostEqual(result["translation_mean_m"], 0.05)
        self.assertAlmostEqual(result["rotation_mean_deg"], 45)
        self.assertEqual(result["recomputed_success_percent"], 50)
        self.assertEqual(result["frames_followed_by_reset"], 1)
        self.assertEqual(analysis.analyze_file(self.path, 0.2, 100)["recomputed_success_percent"], 100)

    def test_reject_wrong_frame(self):
        self.rows[0]["image_index"] = 0
        self.save()
        with self.assertRaises(ValueError):
            analysis.analyze_file(self.path, 0.05, 5)

    def test_reject_reset_segment_error(self):
        self.rows[1]["segment_id"] = 0
        self.save()
        with self.assertRaises(ValueError):
            analysis.analyze_file(self.path, 0.05, 5)

    def test_reject_nonfinite_pose(self):
        self.rows[0]["est_tx_m"] = float("nan")
        self.save()
        with self.assertRaises(ValueError):
            analysis.analyze_file(self.path, 0.05, 5)

    def test_reject_truncated_file(self):
        self.rows.pop()
        self.save()
        with self.assertRaises(ValueError):
            analysis.analyze_file(self.path, 0.05, 5)

    def test_reject_wrong_units(self):
        self.metadata["translation_unit"] = "mm"
        self.save()
        with self.assertRaises(ValueError):
            analysis.analyze_file(self.path, 0.05, 5)

    def test_polar_projection_removes_float_scale_drift(self):
        angle = 0.002
        c, s = math.cos(angle), math.sin(angle)
        expected = [c, -s, 0, s, c, 0, 0, 0, 1]
        projected = analysis.orthogonal_rotation([v * 1.000001 for v in expected])
        for actual, target in zip(projected, expected):
            self.assertAlmostEqual(actual, target, delta=1e-12)

    def test_reject_singular_rotation(self):
        with self.assertRaises(ValueError):
            analysis.orthogonal_rotation([0] * 9)

    def test_cli_rejects_incomplete_run_manifest(self):
        self.save()
        (self.directory / "run_manifest.json").write_text(
            json.dumps({"Status": "running", "ExpectedConfigurations": 1}), encoding="utf-8")
        with patch("sys.argv", ["analyze", "--input", str(self.directory)]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                analysis.main()
        self.assertEqual(raised.exception.code, 1)

    def test_cli_does_not_overwrite_input(self):
        self.save()
        original = self.path.read_bytes()
        with patch("sys.argv", ["analyze", "--input", str(self.path), "--output", str(self.path)]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                analysis.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(self.path.read_bytes(), original)


class ActualExportTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RBOT_POSE_RUN"), "Set RBOT_POSE_RUN to verify actual C++ output")
    def test_ground_truth_alignment_and_pre_reset_estimates(self):
        run = Path(os.environ["RBOT_POSE_RUN"])
        paths = sorted((run / "results" / "poses").glob("poses_*.csv"))
        self.assertTrue(paths)
        for path in paths:
            metadata_path = path.with_name(path.name.replace("poses_", "metadata_", 1)).with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            # This fixture checks the actual RBOT evaluator's fixed camera.
            self.assertEqual(metadata["world_to_camera_row_major"],
                             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
            with Path(metadata["ground_truth_file"]).open() as stream:
                next(stream)
                truth = [[float(v) for v in line.split()] for line in stream if line.strip()]
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            result = analysis.analyze_file(path, 0.05, 5)
            self.assertEqual(len(rows), 1000)
            self.assertLess(result["max_difference_from_native_translation_m"], 2e-6)
            self.assertLess(result["max_difference_from_native_rotation_rad"], 1e-3)
            diagnostics_path = run / "diagnostics" / path.name.replace("poses_", "diagnostics_", 1)
            if diagnostics_path.exists():
                with diagnostics_path.open(newline="") as stream:
                    diagnostics = list(csv.DictReader(stream))
                self.assertEqual([r["tracking_success"] for r in rows], [r["tracking_success"] for r in diagnostics])
            for row in rows:
                gt_rotation, gt_translation = analysis.pose_values(row, "gt")
                original = truth[int(row["image_index"])]
                for actual, expected in zip(gt_rotation + gt_translation, original[:9] + [v * 0.001 for v in original[9:12]]):
                    self.assertAlmostEqual(actual, expected, delta=2e-6)
                if row["tracking_success"] == "0":
                    # If export accidentally moved after ResetBody these would
                    # be equal and failures would masquerade as perfect poses.
                    self.assertNotEqual(analysis.pose_values(row, "est"), (gt_rotation, gt_translation))


if __name__ == "__main__":
    unittest.main()
