"""Protocol tests, including missing frames, metre GT and nonfinite estimates.

Run: python -m unittest discover -s test -p test_industrial_adapter.py -v
"""
import csv
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import shutil
import types
import unittest
from unittest.mock import patch
from pathlib import Path

HELPERS = Path(__file__).resolve().parents[1] / "examples/industrial"
sys.path.insert(0, str(HELPERS))
import dataset
import install
import metrics
import run

IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]


class Fixture(unittest.TestCase):
    def setUp(self):
        # Keep paths short on Windows while preserving normal temporary cleanup.
        self.temp = tempfile.TemporaryDirectory(prefix="m3ti_")
        self.root = Path(self.temp.name).resolve()
        self.config = dataset.configurations(["scene_01"], ["camera_motion"], ["01_normal"], ["obj_000001"])[0]
        self.images, self.poses, self.model = dataset.paths(self.root, self.config)
        self.images.mkdir(parents=True)
        self.poses.parent.mkdir(parents=True)
        self.model.parent.mkdir(parents=True)
        self.model.write_text("mtllib obj_000001.mtl\nv 0 0 0\n", encoding="utf-8")
        self.model.with_suffix(".mtl").write_text("newmtl m\nmap_Kd diffuse.jpg\n", encoding="utf-8")
        (self.model.parent / "diffuse.jpg").write_bytes(b"fixture diffuse map")
        self.poses.write_text("#r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz\n" +
                              (" ".join(map(str, IDENTITY+[0, 0, 2.5]))+"\n")*500, encoding="utf-8")
        camera = {"width": 900, "height": 600, "K": [[1125, 0, 449.5], [0, 1125, 299.5], [0, 0, 1]],
                  "frames_per_sequence": 500, "distortion_coefficients": [0]*5, "translation_unit": "meter",
                  "pose_direction": "p_camera = R @ p_object + t", "motion_modes": dataset.MOTIONS,
                  "conditions": dataset.CONDITIONS, "models": [{"object_id": i, "units": "meter"} for i in range(1, 16)]}
        (self.root / "cameras").mkdir()
        dataset.write_json(self.root / "cameras/camera.json", camera)
        for i in range(1, 4):
            (self.images / ("%06d.png" % i)).write_bytes(self.png())
        self.output = self.root / "output"
        self.output.mkdir()
        self.csv = self.output / "results/poses/poses_fixture.csv"
        self.csv.parent.mkdir(parents=True)
        self.metadata = dict(schema_version=1, dataset_kind="industrial", object="obj_000001",
                             sequence="scene_01__camera_motion__01_normal", translation_unit="m",
                             model_unit_in_meter=1, ground_truth_translation_scale=1,
                             pose_convention="p_camera = R * p_body + t", initial_image_index=1,
                             first_image_index=2, last_image_index=3, frame_count=2,
                             reset_protocol="industrial_selected_output_reset",
                             estimate_capture="final_selected_output_before_gt_reset",
                             input_conversion="uint16_to_uint8_fixed_1_over_257",
                             translation_threshold_m=.05, rotation_threshold_rad=math.radians(5),
                             intrinsics={"fx": 1125, "fy": 1125, "cx": 449.5, "cy": 299.5, "width": 900, "height": 600},
                             initial_gt_pose_camera_row_major=[1,0,0,0,0,1,0,0,0,0,1,2.5,0,0,0,1])
        dataset.write_json(self.csv.with_name("metadata_fixture.json"), self.metadata)
        overlay = self.output / "results/overlays/initial_gt_image_1.png"
        overlay.parent.mkdir()
        overlay.write_bytes(b"fixture_overlay"*10)
        self.rows = []
        for frame in range(2):
            row = dict(frame=frame, image_index=frame+2, image_file="%06d.png" % (frame+2), segment_id=0,
                       estimate_source="single", pose_finite=1, reset_primary_after_frame=0,
                       reset_secondary_after_frame=0, tracking_success=1, translation_error_m=0,
                       rotation_error_rad=0, cycle_us=1000)
            for prefix in ("est", "gt"):
                row.update({"%s_r%d%d" % (prefix, r, c): IDENTITY[r*3+c] for r in range(3) for c in range(3)})
                row.update({prefix+"_tx_m": 0, prefix+"_ty_m": 0, prefix+"_tz_m": 2.5})
            self.rows.append(row)
        self.save_rows()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def png(bits=16, width=900):
        return b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+struct.pack(">IIBBBBB",width,600,bits,2,0,0,0)+b"\0"*8

    def save_rows(self):
        with self.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def analyze(self):
        return metrics.analyze(self.output, self.root, self.config, 3)


class ProtocolTests(Fixture):
    def test_full_denominator_and_scene_membership(self):
        configs = dataset.configurations()
        self.assertEqual(len(configs), 150)
        self.assertEqual(len({c["id"] for c in configs}), 150)
        self.assertEqual(150*499, 74850)
        with self.assertRaises(ValueError):
            dataset.configurations(["scene_01"], objects=["obj_000015"])

    def test_preflight_preserves_metre_poses(self):
        before = self.poses.read_bytes()
        report = dataset.preflight(self.root, [self.config], 3)
        self.assertEqual(report["evaluated_frames"], 2)
        self.assertEqual(report["unique_images_checked"], 3)
        self.assertEqual(dataset.read_poses(self.poses)[0][-1], 2.5)
        self.assertEqual(self.poses.read_bytes(), before)

    def test_missing_image_fails_instead_of_reusing_previous(self):
        (self.images / "000003.png").unlink()
        with self.assertRaises(OSError):
            dataset.preflight(self.root, [self.config], 3)

    def test_wrong_png_type_and_calibration_rejected(self):
        image = self.images / "000001.png"
        image.write_bytes(self.png(8))
        with self.assertRaisesRegex(ValueError, "16-bit RGB"):
            dataset.png_header(image)
        camera = json.loads((self.root / "cameras/camera.json").read_text())
        camera["K"][0][2] = 449.75
        dataset.write_json(self.root / "cameras/camera.json", camera)
        with self.assertRaises(ValueError):
            dataset.read_camera(self.root)

    def test_incomplete_pose_file_rejected(self):
        self.poses.write_text("1 0 0 0 1 0 0 0 1 0 0 2.5\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "500 poses"):
            dataset.read_poses(self.poses)

    def test_native_and_recomputed_identity(self):
        report = self.analyze()
        self.assertEqual(report["success_percent"], 100)
        self.assertEqual(report["recomputed_success_percent"], 100)
        self.assertEqual(report["frames"], 2)

    def test_failure_then_next_segment(self):
        self.rows[0].update(est_tx_m=.051, translation_error_m=.051, tracking_success=0, reset_primary_after_frame=1)
        self.rows[1].update(segment_id=1, est_tx_m=.049, translation_error_m=.049)
        self.save_rows()
        report = self.analyze()
        self.assertEqual(report["success_percent"], 50)
        self.assertEqual(report["recomputed_success_percent"], 50)
        self.assertEqual(report["reset_count"], 1)

    def test_nonfinite_estimate_counts_failure_without_dropping(self):
        self.rows[0].update(est_tx_m="nan", pose_finite=0, tracking_success=0,
                            reset_primary_after_frame=1, translation_error_m="inf", rotation_error_rad="inf")
        self.rows[1]["segment_id"] = 1
        self.save_rows()
        report = self.analyze()
        self.assertEqual(report["frames"], 2)
        self.assertEqual(report["success_percent"], 50)
        self.assertEqual(report["invalid_estimate_count"], 1)
        self.assertEqual(report["error_statistics_frame_count"], 1)
        self.assertEqual(report["nonfinite_native_error_images"], [2])

    def test_singular_estimate_is_also_failure(self):
        self.rows[0].update(est_r00=0, tracking_success=0, reset_primary_after_frame=1,
                            translation_error_m="inf", rotation_error_rad="inf")
        self.rows[1]["segment_id"] = 1
        self.save_rows()
        self.assertEqual(self.analyze()["invalid_estimate_count"], 1)

    def test_gt_unit_or_row_shift_rejected(self):
        self.rows[0]["gt_tz_m"] = .0025
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "GT differs"):
            self.analyze()

    def test_broken_segment_or_partial_csv_rejected(self):
        self.rows[1]["segment_id"] = 1
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "segment"):
            self.analyze()
        self.rows = self.rows[:1]
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "Incomplete CSV"):
            self.analyze()

    def test_summary_is_explicit_about_subset_and_denominator(self):
        result = run.summarize(self.output, [self.analyze()], "selected subset / smoke test")
        self.assertEqual(result["scope"], "selected subset / smoke test")
        self.assertEqual(result["overall"]["frames"], 2)
        self.assertTrue((self.output / "success_rates.csv").exists())
        self.assertTrue((self.output / "results.md").exists())


class InstallerTests(unittest.TestCase):
    def test_unknown_changes_are_rejected_before_any_write_and_crlf_is_preserved_in_backup(self):
        with tempfile.TemporaryDirectory(prefix="m3tii_") as temporary:
            root = Path(temporary)
            kit, source = root / "kit", root / "source"
            source.mkdir()
            (kit / "payload").mkdir(parents=True)
            files = {}
            for name in ("first.cpp", "second.cpp"):
                (source / name).write_bytes(b"original\r\n")
                (kit / "payload" / name).write_bytes(b"new\n")
                files[name] = dict(new_sha256=install.digest(b"new\n"),
                                   allowed_base_sha256=[install.digest(b"original\n")], allow_missing=False)
            dataset.write_json(kit / "payload_manifest.json", {"files": files})
            (source / "second.cpp").write_bytes(b"user change\n")
            with self.assertRaisesRegex(ValueError, "Local changes"):
                install.install(kit, source, require_linux=False)
            self.assertEqual((source / "first.cpp").read_bytes(), b"original\r\n")
            self.assertFalse((source / "industrial-eval-backups").exists())
            (source / "second.cpp").write_bytes(b"original\r\n")
            install.install(kit, source, require_linux=False)
            backups = list((source / "industrial-eval-backups").glob("*/first.cpp"))
            self.assertEqual(backups[0].read_bytes(), b"original\r\n")
            self.assertEqual(install.install(kit, source, require_linux=False), [])


class RunnerTests(Fixture):
    def test_end_to_end_orchestration_and_resume_reaudits_without_rerunning(self):
        source = self.root / "source"
        binary = source / "build-linux/examples/evaluate_industrial_dataset"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"fixture binary")
        model = source / "disk_512.onnx"
        model.write_bytes(b"fixture model")
        output = self.root / "run"
        cli = ["run.py", "--source", str(source), "--dataset", str(self.root), "--output", str(output),
               "--cache", str(self.root / "cache"), "--frames", "3", "--scene", "scene_01",
               "--motion", "camera_motion", "--condition", "01_normal", "--object", "obj_000001"]
        invocations = []

        def execute(command, **kwargs):
            if command[0] == "xvfb-run":
                self.assertEqual(command[command.index("--frames")+1], "3")
                target = Path(command[command.index("--output")+1])
                shutil.copytree(self.output / "results", target / "results", dirs_exist_ok=True)
                invocations.append(command)
            return types.SimpleNamespace(returncode=0)

        fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with patch.dict(sys.modules, {"fcntl": fcntl}), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-fixture"}), \
                patch.object(run, "DISK_SHA256", dataset.sha256(model)), patch.object(run.shutil, "which", return_value="xvfb-run"), \
                patch.object(run.os, "access", return_value=True), patch.object(run.subprocess, "run", side_effect=execute):
            with patch.object(sys, "argv", cli):
                run.main()
            manifest_path = output / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["expected_evaluated_frames"], 2)
            self.assertEqual(len(invocations), 1)
            manifest["status"] = "failed"
            dataset.write_json(manifest_path, manifest)
            with patch.object(sys, "argv", cli+["--resume"]):
                run.main()
            self.assertEqual(len(invocations), 1)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["status"], "complete")
            # A changed binary cannot mix its metrics into an existing run.
            binary.write_bytes(b"different algorithm binary")
            with patch.object(sys, "argv", cli+["--resume"]), self.assertRaisesRegex(SystemExit, "identity differs"):
                run.main()


if __name__ == "__main__":
    unittest.main()
