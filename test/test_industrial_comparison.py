"""Paired benchmark orchestration and timing contracts (mock tracker outputs)."""
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import time

from test_industrial_adapter import Fixture, dataset, metrics

ADAPTER = Path(__file__).resolve().parents[1]/"examples/industrial_compare"
sys.path.insert(0, str(ADAPTER))
import compare
import report
import provenance


class ComparisonFixture(Fixture):
    def setUp(self):
        super().setUp()
        for row in self.rows:
            row.update(decode_us=100, preprocessing_us=10, tracking_wall_us=1200, compute_wall_us=1100,
                       metric_us=20, reset_us=0, frame_wall_us=1220, before_camera_us=0,
                       correspondences_us=800, gradient_hessian_us=100, optimization_us=50, results_us=50)
        self.save_rows()

    def args(self, output=None):
        return compare.parse_args(["--dataset", str(self.root), "--output", str(output or self.root/"run"),
             "--cache", str(self.root/"cache"), "--frames", "3", "--warmup", "1", "--methods", "rbgt", "srt3d",
             "--scene", "scene_01", "--motion", "camera_motion", "--condition", "01_normal", "--object", "obj_000001"])


class TimingTests(ComparisonFixture):
    def test_warmup_excluded_only_from_time_and_paired_denominators(self):
        self.rows[0].update(est_tx_m=.051, translation_error_m=.051, tracking_success=0, reset_primary_after_frame=1)
        self.rows[1]["segment_id"] = 1
        self.save_rows()
        items = []
        for method in ("rbgt", "srt3d"):
            item = self.analyze(); item.update(method=method, repeat=1); items.append(item)
        identity = dict(methods=["rbgt", "srt3d"], configurations=[self.config], repeats=1)
        result = report.summarize(self.root, items, 1, identity, "smoke")
        self.assertEqual(result["methods"]["rbgt"]["success_percent"], 50)
        self.assertEqual(result["methods"]["rbgt"]["frames"], 2)
        self.assertEqual(result["methods"]["rbgt"]["timing"]["frame_wall_us"]["frames"], 1)
        self.assertEqual(result["primary_timing_metric"], "compute_wall_us")
        self.assertAlmostEqual(result["methods"]["rbgt"]["derived_compute_fps"], 1e6/1100)
        text = (self.root/"results.md").read_text(encoding="utf-8")
        self.assertIn("每帧计算均值", text)
        self.assertNotIn("完整帧均值 ms", text)
        self.assertEqual(len(result["paired_configurations"]), 1)
        self.assertTrue((self.root/"results.md").is_file())

    def test_invalid_time_or_bad_decomposition_rejected(self):
        self.rows[0]["frame_wall_us"] = 500
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "Frame timing"):
            report.timing_rows(self.csv, 0)
        self.rows[0]["decode_us"] = float("nan")
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            report.timing_rows(self.csv, 0)

    def test_missing_method_prevents_complete_report(self):
        item = self.analyze(); item.update(method="rbgt", repeat=1)
        identity = dict(methods=["rbgt", "srt3d"], configurations=[self.config], repeats=1)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            report.summarize(self.root, [item], 0, identity, "smoke")
        partial = report.summarize(self.root, [item], 0, identity, "smoke", partial=True)
        self.assertEqual(partial["status"], "partial")

    def test_v09_single_motion_supported_without_weakening_old_contract(self):
        camera_path = self.root/"cameras/camera.json"
        camera = json.loads(camera_path.read_text())
        camera["motion_modes"] = ["camera_and_objects_motion"]
        dataset.write_json(camera_path, camera)
        with self.assertRaises(ValueError):
            dataset.read_camera(self.root)
        selected = dataset.read_camera(self.root, allow_motion_subset=True)
        configs = dataset.configurations(motions=selected["motion_modes"])
        self.assertEqual(len(configs), 75)
        self.assertEqual(len(configs)*499, 37425)
        with self.assertRaisesRegex(ValueError, "not declared"):
            dataset.preflight(self.root, [self.config], 3, allow_motion_subset=True)

    def test_method_order_rotates_without_duplicating_jobs(self):
        configs = dataset.configurations()
        jobs = list(compare.jobs(configs, report.METHODS, 2))
        self.assertEqual(len(jobs), 1500)
        self.assertEqual(len({j["key"] for j in jobs}), 1500)
        self.assertEqual(jobs[0]["method"], "rbgt")
        self.assertEqual(jobs[5]["method"], "srt3d")
        self.assertEqual(jobs[750]["method"], "srt3d")


class RunnerTests(ComparisonFixture):
    def test_native_crash_displays_log_tail_and_path(self):
        args = self.args()
        def fake_run(command, **kwargs):
            if "--output" in command:
                kwargs["stdout"].write("BodyData does not fit into image\nAborted\n")
                kwargs["stdout"].flush()
                raise subprocess.CalledProcessError(134, command)
        with patch.object(compare, "hold_lock"), patch.object(compare, "protect_dataset"), \
             patch.object(compare, "verified_build", return_value={"binary_sha256": "test"}), \
             patch.object(compare.shutil, "which", return_value="xvfb-run"), \
             patch.object(compare.subprocess, "run", side_effect=fake_run):
            with contextlib.ExitStack() as resources, self.assertRaisesRegex(RuntimeError, "(?s)status 134.*evaluation.log.*BodyData does not fit"):
                compare.execute(args, resources)

    def test_run_and_resume_reaudit_and_reject_changed_data(self):
        args = self.args(); evaluations = []
        def fake_run(command, **kwargs):
            if "--output" in command:
                directory = Path(command[command.index("--output")+1])
                evaluations.append(command)
                shutil.copytree(self.output/"results", directory/"results", dirs_exist_ok=True)
        fake_build = lambda kit, method: {"method": method, "binary_sha256": method+"1234"}
        with patch.object(compare, "hold_lock"), patch.object(compare, "protect_dataset"), \
             patch.object(compare, "verified_build", side_effect=fake_build), \
             patch.object(compare.shutil, "which", return_value="xvfb-run"), \
             patch.object(compare.subprocess, "run", side_effect=fake_run):
            with contextlib.ExitStack() as resources:
                compare.execute(args, resources)
            self.assertEqual(len(evaluations), 2)
            summary = json.loads((args.output/"summary.json").read_text())
            self.assertEqual(summary["methods"]["rbgt"]["frames"], 2)
            args.resume = True
            with contextlib.ExitStack() as resources:
                compare.execute(args, resources)
            self.assertEqual(len(evaluations), 2)
            (self.images/"000003.png").write_bytes(self.png()+b"changed")
            with contextlib.ExitStack() as resources, self.assertRaisesRegex(ValueError, "Dataset changed"):
                compare.execute(args, resources)
            manifest = json.loads((args.output/"run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_failure_keeps_job_incomplete_and_reports_failed(self):
        args = self.args()
        def fake_run(command, **kwargs):
            if "--output" in command:
                raise RuntimeError("mock tracker failure")
        with patch.object(compare, "hold_lock"), patch.object(compare, "protect_dataset"), \
             patch.object(compare, "verified_build", return_value={"binary_sha256": "test"}), \
             patch.object(compare.shutil, "which", return_value="xvfb-run"), \
             patch.object(compare.subprocess, "run", side_effect=fake_run):
            with contextlib.ExitStack() as resources, self.assertRaisesRegex(RuntimeError, "mock tracker"):
                compare.execute(args, resources)
        manifest = json.loads((args.output/"run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_jobs"], [])
        self.assertFalse((args.output/"summary.json").exists())

    def test_build_fingerprint_detects_source_tampering(self):
        kit = self.root/"kit"
        source = kit/"sources/upstream/RBGT/src"; source.mkdir(parents=True)
        (source/"CMakeLists.txt").write_text("source list")
        (source/"tracker.cpp").write_text("original")
        (kit/"CMakeLists.txt").write_text("wrapper")
        (kit/"baseline.cpp").write_text("adapter")
        build = kit/"build/rbgt"; build.mkdir(parents=True)
        binary = build/"evaluate_industrial_baseline"; binary.write_bytes(b"binary")
        cache = build/"CMakeCache.txt"; cache.write_text("release")
        dataset.write_json(build/"build_identity.json", {"method": "rbgt", "binary_sha256": dataset.sha256(binary),
             "source_sha256": provenance.source_files(kit, "rbgt"), "cmake_cache_sha256": dataset.sha256(cache)})
        provenance.verified_build(kit, "rbgt")
        (source/"tracker.cpp").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Build/source changed"):
            provenance.verified_build(kit, "rbgt")
        # A tar upload may preserve an older mtime than the server's objects.
        # Changed content must still cause a rebuild; unchanged inputs stay put.
        changed_source = source/"tracker.cpp"
        os.utime(str(changed_source), (1, 1))
        before = time.time()-1
        changed = provenance.prepare(kit, "rbgt")
        self.assertEqual(changed, ["source/src/tracker.cpp"])
        self.assertGreater(changed_source.stat().st_mtime, before)
        self.assertEqual(changed_source.read_text(), "tampered")


class LegacyProjectionTests(unittest.TestCase):
    def test_actual_cpp_bounds_function_encloses_perspective_sphere(self):
        compiler = shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler required for geometry regression")
        code = (ADAPTER/"baseline.cpp").read_text(encoding="utf-8")
        function = "float LegacyModelDiameter("+code.split("float LegacyModelDiameter(", 1)[1].split("template<class F>", 1)[0]
        template = Path(__file__).with_name("test_legacy_model_bounds.cpp").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="m3t_bounds_") as directory:
            source = Path(directory)/"bounds.cpp"
            binary = Path(directory)/("bounds.exe" if os.name == "nt" else "bounds")
            source.write_text(template.replace("// LEGACY_BOUNDS_IMPLEMENTATION", function), encoding="utf-8")
            subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("regression passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
