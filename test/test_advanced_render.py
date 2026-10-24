"""Advanced-v1 input contracts and compatibility with existing tracker exports."""
import contextlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from test_industrial_adapter import Fixture, dataset
ADAPTER = Path(__file__).resolve().parents[1]/"examples/industrial_render_advanced"
sys.path.insert(0, str(ADAPTER))
import audit_inputs
spec = importlib.util.spec_from_file_location("advanced_evaluate", ADAPTER/"evaluate.py")
evaluate = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)
BASH = os.environ.get("M3T_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)


@unittest.skipUnless(BASH, "Set M3T_TEST_BASH to a native Bash executable on Windows")
class RenderGuardTests(unittest.TestCase):
    def run_guard(self, script, allow=None):
        environment = os.environ.copy()
        environment.pop("ADVANCED_ALLOW_CONCURRENT", None)
        if allow is not None:
            environment["ADVANCED_ALLOW_CONCURRENT"] = allow
        # Simulate a held lock without depending on Windows flock support.
        prefix = 'source "$1"; BASE="$2"; M3T_COMPARE_CACHE="$BASE/cache"; flock() { return 1; }; '
        with tempfile.TemporaryDirectory(prefix="advanced_guard_") as directory:
            return subprocess.run([BASH, "-c", prefix + script, "--", (ADAPTER/"guard.sh").as_posix(),
                                   Path(directory).as_posix()], env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_default_rejects_busy_benchmark(self):
        result = self.run_guard("guard_benchmark")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Five-method evaluation is active", result.stdout)

    def test_opt_in_permits_benchmark_overlap_but_keeps_workspace_guard(self):
        result = self.run_guard("guard_benchmark; printf 'benchmark_overlap_allowed\\n'; guard_workspace", "1")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("benchmark_overlap_allowed", result.stdout)
        self.assertIn("preparation/render task is active", result.stdout)
        self.assertNotIn("Five-method evaluation is active", result.stdout)

    def test_invalid_concurrency_flag_is_rejected(self):
        result = self.run_guard("guard_benchmark", "yes")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("must be 0 or 1", result.stdout)


@unittest.skipUnless(BASH, "Set M3T_TEST_BASH to a native Bash executable on Windows")
class RenderResourceTests(unittest.TestCase):
    def run_limited(self, mode, cpus=None):
        with tempfile.TemporaryDirectory(prefix="advanced_resources_") as directory:
            root = Path(directory); binaries = root/"bin"; binaries.mkdir(); log = root/"calls.txt"
            kit = root/"kit"; kit.mkdir(); shutil.copyfile(ADAPTER/"render_limited.sh", kit/"render_limited.sh")
            with (kit/"render.sh").open("w", encoding="utf-8", newline="\n") as stream:
                stream.write('printf "render|%s\\n" "$1" >> "$M3T_LIMIT_TEST_LOG"\n'
                             'printf "BLAS=%s,%s\\n" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" >> "$M3T_LIMIT_TEST_LOG"\n')
            with (kit/"prepare.sh").open("w", encoding="utf-8", newline="\n") as stream:
                stream.write('printf "prepare|%s\\n" "$#" >> "$M3T_LIMIT_TEST_LOG"\n'
                             'printf "BLAS=%s,%s\\n" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" >> "$M3T_LIMIT_TEST_LOG"\n')
            # The Windows host cannot exercise Linux scheduling. These stubs
            # verify the complete launch chain without launching a real render.
            for name in ("taskset", "nice", "ionice"):
                # Git Bash's /bin/sh can prepend its system tools to PATH.
                text = '#!/bin/sh\nPATH="${0%/*}:$PATH"; export PATH\n'
                text += "{ printf '%s' '"+name+"'; printf '|%s' \"$@\"; printf '\\n'; } >> \"$M3T_LIMIT_TEST_LOG\"\n"
                text += 'shift 2\nexec "$@"\n'
                executable = binaries/name
                with executable.open("w", encoding="utf-8", newline="\n") as stream: stream.write(text)
                executable.chmod(0o755)
            environment = os.environ.copy(); environment.pop("ADVANCED_RENDER_CPUS", None)
            if cpus is not None: environment["ADVANCED_RENDER_CPUS"] = cpus
            environment["PATH"] = str(binaries)+os.pathsep+environment.get("PATH", "")
            environment["M3T_LIMIT_TEST_LOG"] = log.as_posix()
            result = subprocess.run([BASH, (kit/"render_limited.sh").as_posix(), mode], env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return result, log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def test_full_launch_inherits_affinity_and_advisory_priorities(self):
        result, calls = self.run_limited("full", "24-31,72-79")
        self.assertEqual(result.returncode, 0, result.stdout+repr(calls))
        self.assertEqual(len(calls), 5, calls)
        self.assertTrue(calls[0].startswith("taskset|--cpu-list|24-31,72-79|nice|-n|15|ionice|-c|3|bash|"), calls)
        self.assertTrue(calls[1].startswith("nice|-n|15|ionice|-c|3|bash|"), calls)
        self.assertTrue(calls[2].startswith("ionice|-c|3|bash|"), calls)
        self.assertEqual(calls[3], "render|full")
        self.assertEqual(calls[4], "BLAS=1,1")

    def test_status_bypasses_resource_requirements(self):
        result, calls = self.run_limited("status")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(calls[0], "render|status")
        self.assertEqual(len(calls), 2, calls)

    def test_preparation_uses_the_same_limits_without_render_arguments(self):
        result, calls = self.run_limited("prepare", "24-31,72-79")
        self.assertEqual(result.returncode, 0, result.stdout+repr(calls))
        self.assertEqual(len(calls), 5, calls)
        self.assertTrue(calls[0].startswith("taskset|--cpu-list|24-31,72-79|nice|-n|15|ionice|-c|3|bash|"), calls)
        self.assertEqual(calls[3], "prepare|0")

    def test_missing_or_malformed_cpu_selection_never_launches(self):
        for cpus in (None, "24-31;echo invalid"):
            result, calls = self.run_limited("full", cpus)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("Set ADVANCED_RENDER_CPUS", result.stdout)
            self.assertEqual(calls, [])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.occlusion = np.zeros((500, 3)); self.truncation = np.zeros((500, 3))
        self.occlusion[110:150, 1] = .4; self.occlusion[310:350, 1] = .4

    def test_plateau_indices_are_one_based_and_checked_at_both_events(self):
        result = audit_inputs.check_events(self.occlusion, self.truncation, 1, "O02")
        self.assertAlmostEqual(result["plateau_min"], .4)
        self.occlusion[349, 1] = .1
        with self.assertRaisesRegex(ValueError, "plateau"):
            audit_inputs.check_events(self.occlusion, self.truncation, 1, "O02")

    def test_off_event_and_focus_truncation_are_not_silently_accepted(self):
        self.occlusion[99, 1] = .06
        with self.assertRaisesRegex(ValueError, "Unscheduled"):
            audit_inputs.check_events(self.occlusion, self.truncation, 1, "O02")
        self.occlusion[99, 1] = 0; self.truncation[123, 1] = .01
        with self.assertRaisesRegex(ValueError, "out of frame"):
            audit_inputs.check_events(self.occlusion, self.truncation, 1, "O02")

    def test_world_rotations_reject_scale_and_reflection(self):
        matrices = np.tile(np.eye(4), (3, 1, 1))
        audit_inputs.check_se3(matrices, "fixture")
        matrices[1, 0, 0] = 1.1
        with self.assertRaises(ValueError): audit_inputs.check_se3(matrices, "fixture")
        matrices[1, 0, 0] = -1
        with self.assertRaises(ValueError): audit_inputs.check_se3(matrices, "fixture")

    def test_index_counts_and_focus_selection(self):
        rows = []
        for scene, ids in audit_inputs.SCENES.items():
            for t in audit_inputs.TRAJECTORIES:
                for c in audit_inputs.CONDITIONS:
                    for oid in ids:
                        rows.append(dict(scene=scene, trajectory=t, condition=c,
                                         is_focus=oid == ids[int(t[-1])-1]))
        index = {"configurations": rows}
        self.assertEqual(len(evaluate.select_rows(index)), 324)
        self.assertEqual(len(evaluate.select_rows(index, True)), 108)
        self.assertEqual(len(evaluate.select_rows(index, smoke=True)), 3)
        self.assertEqual(324*499, 161676); self.assertEqual(108*499, 53892)


class AdvancedRunnerTests(Fixture):
    def test_native_alias_audited_before_restoring_real_condition_and_resume(self):
        sys.path.insert(0, str(ADAPTER.parent/"industrial_compare"))
        import compare, provenance
        for row in self.rows:
            row.update(decode_us=100, preprocessing_us=10, tracking_wall_us=1200, compute_wall_us=1100,
                       metric_us=20, reset_us=0, frame_wall_us=1220, before_camera_us=0,
                       correspondences_us=800, gradient_hessian_us=100, optimization_us=50, results_us=50)
        self.save_rows()
        root = self.root/"advanced"; data = root/"交付物"/evaluate.NAME; data.mkdir(parents=True)
        state = root/"中间结果文件/run_state_advanced_v1"
        (state/"validation").mkdir(parents=True); (state/"independent_audit").mkdir()
        dataset.write_json(state/"validation/final_audit.json", {"status": "passed", "scope": "complete_dataset", "images": 54000})
        dataset.write_json(state/"independent_audit/inputs.json", {"status": "passed"})
        dataset.write_json(state/"independent_audit/scenes.json", {"status": "passed", "scope": "all integer frames",
                           "sequences_checked": 108, "total_frame_checks": 54000})
        config = dict(sequence_id="supp_scene_01_t01_R03", scene="scene_01", motion_mode="camera_motion",
                      trajectory="t01", condition="R03", object="obj_000001", is_focus=True,
                      pair_id="scene_01_t01", split="development")
        dataset.write_json(state/"independent_audit/evaluation_index.json", {"configurations": [config]})
        calls = []
        def fake_view(source, row, view):
            for name in ("images", "poses", "objects", "cameras"):
                shutil.copytree(self.root/name, view/name, dirs_exist_ok=True)
            return evaluate.native_config(row)
        def fake_run(cmd, **kwargs):
            destination = Path(cmd[cmd.index("--output")+1]); calls.append(cmd)
            self.assertEqual(cmd[cmd.index("--condition")+1], "01_normal")
            shutil.copytree(self.output/"results", destination/"results", dirs_exist_ok=True)
        kit = ADAPTER.parent/"industrial_compare"; out = self.root/"new_run"
        args = ["--root", str(root), "--kit", str(kit), "--output", str(out), "--frames", "3", "--warmup", "1"]
        with patch.object(compare, "hold_lock"), patch.object(provenance, "verified_build", return_value={"binary_sha256": "mock"}), \
             patch.object(dataset, "sha256", return_value=dataset.DISK_SHA256), \
             patch.object(evaluate.shutil, "which", return_value="xvfb-run"), \
             patch.object(evaluate, "create_view", side_effect=fake_view), \
             patch.object(evaluate, "input_identity", return_value="test_input"), \
             patch.object(evaluate.subprocess, "run", side_effect=fake_run), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test", "M3T_COMPARE_CACHE": str(self.root/"cache")}):
            evaluate.main(args)
            self.assertEqual(len(calls), 5)
            report = json.loads((out/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["configurations"][0]["condition"], "R03")
            self.assertEqual(report["configurations"][0]["trajectory"], "t01")
            self.assertEqual(report["methods"]["ours"]["frames"], 2)
            self.assertEqual(report["identity"]["method_labels"]["icg_rgb"], "ICG")
            self.assertEqual(report["identity"]["method_labels"]["m3t_original_rgb"], "ICG+")
            self.assertTrue((out/"focus/summary.json").exists())
            evaluate.main(["--root", str(root), "--kit", str(kit), "--output", str(out), "--resume"])
            self.assertEqual(len(calls), 5)

    def test_sampled_scene_audit_cannot_start_native_evaluation(self):
        sys.path.insert(0, str(ADAPTER.parent/"industrial_compare"))
        import compare
        audits = [{"status": "passed", "scope": "complete_dataset", "images": 54000}, {"status": "passed"},
                  {"status": "passed", "scope": "sampled integer frames", "sequences_checked": 108, "total_frame_checks": 1080}]
        with patch.object(compare, "hold_lock"), patch.object(evaluate, "read_json", side_effect=audits), \
             patch.object(evaluate.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "independent Blender"):
                evaluate.main(["--root", str(self.root), "--kit", str(ADAPTER.parent/"industrial_compare"),
                               "--output", str(self.root/"rejected_run")])
            execute.assert_not_called()


if __name__ == "__main__": unittest.main()
