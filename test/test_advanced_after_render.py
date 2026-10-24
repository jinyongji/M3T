"""A queued evaluation must never consume partial rendering or bypass a failed smoke run."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

ADAPTER = Path(__file__).parent.parent/'examples/industrial_render_advanced'
spec = importlib.util.spec_from_file_location('queued_advanced_evaluation', ADAPTER/'after_render.py')
queued = importlib.util.module_from_spec(spec); spec.loader.exec_module(queued)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='advanced_queue_')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory/'dataset'; self.root.mkdir()
        self.state = self.root/'中间结果文件/run_state_advanced_v1'; self.state.mkdir(parents=True)
        self.output = self.directory/'run'
        self.idle = dict(uuid='GPU-test', memory_used_mib=24, utilization_percent=0, compute_pids=[])

    def main_patches(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}))
        stack.enter_context(patch.dict('sys.modules', {'fcntl': SimpleNamespace(
            LOCK_EX=2, LOCK_SH=1, LOCK_NB=4, flock=Mock())}))
        return stack

    def args(self):
        return ['--root', str(self.root), '--output', str(self.output), '--poll-seconds', '1']

    def test_rendering_and_partial_final_audit_are_not_complete(self):
        queued.write_json(self.state/'status.json', {'status': 'rendering', 'job': 28, 'pid': os.getpid()})
        with patch.object(queued, 'controller_alive', return_value=True):
            self.assertFalse(queued.rendering_complete(self.state)[0])
        queued.write_json(self.state/'status.json', {'status': 'complete'})
        with self.assertRaisesRegex(RuntimeError, 'missing'): queued.rendering_complete(self.state)
        queued.write_json(self.state/'validation/final_audit.json', {'status': 'passed', 'scope': 'smoke_test', 'images': 36})
        with self.assertRaisesRegex(RuntimeError, '54,000'): queued.rendering_complete(self.state)
        queued.write_json(self.state/'validation/final_audit.json', {'status': 'passed', 'scope': 'complete_dataset', 'images': 54000})
        self.assertTrue(queued.rendering_complete(self.state)[0])

    def test_exited_or_zombie_render_controller_is_not_treated_as_running(self):
        with patch.object(queued.Path, 'read_text', return_value='42 (blender worker) Z 1 2 3'):
            self.assertFalse(queued.controller_alive(42))
        queued.write_json(self.state/'status.json', {'status': 'rendering', 'pid': 42})
        with patch.object(queued, 'controller_alive', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'controller exited'): queued.rendering_complete(self.state)

    def test_gpu_queries_reject_compute_context_even_if_utilization_is_zero(self):
        outputs = [SimpleNamespace(stdout='GPU-test, 24, 0\n'),
                   SimpleNamespace(stdout='GPU-other, 100\nGPU-test, 200\n')]
        with patch.object(queued.subprocess, 'run', side_effect=outputs): snapshot = queued.gpu_snapshot('GPU-test')
        self.assertEqual(snapshot['compute_pids'], ['200'])
        self.assertFalse(queued.gpu_idle(snapshot))
        self.assertTrue(queued.gpu_idle(self.idle))

    def test_independent_audits_do_not_overwrite_renderer_status(self):
        blender = self.root/'中间结果文件/runtime_advanced_v1/extracted/blender-4.5.4-linux-x64/blender'
        blender.parent.mkdir(parents=True); blender.touch(); self.output.mkdir()
        queued.write_json(self.state/'status.json', {'status': 'complete', 'pid': 123})
        before = (self.state/'status.json').read_bytes(); commands = []
        def fake_run(command, log, environment):
            commands.append(command); destination = Path(command[command.index('--output')+1])
            if destination.name == 'inputs.json':
                queued.write_json(destination, {'status': 'passed', 'pose_files': 324, 'sequences': 108})
                queued.write_json(destination.with_name('evaluation_index.json'), {
                    'configurations': [{'is_focus': i < 108} for i in range(324)]})
            else:
                queued.write_json(destination, {'status': 'passed', 'scope': 'all integer frames',
                                               'sequences_checked': 108, 'total_frame_checks': 54000})
        with patch.object(queued, 'run_logged', side_effect=fake_run): queued.run_independent_audits(self.root, self.output)
        self.assertEqual((self.state/'status.json').read_bytes(), before)
        self.assertEqual(len(commands), 2)

    def test_successful_handoff_waits_then_audits_then_smoke_then_full(self):
        order = []
        busy = dict(self.idle, compute_pids=['200'], utilization_percent=98)
        snapshots = [busy, self.idle, self.idle, self.idle, self.idle]
        def gpu(device):
            value = snapshots.pop(0); order.append('busy' if value['compute_pids'] else 'idle'); return value
        def evaluation(root, output, kit, disk, smoke): order.append('smoke' if smoke else 'full')
        with self.main_patches(), patch.object(queued.time, 'sleep') as sleep, \
             patch.object(queued, 'rendering_complete', side_effect=[(False, {'job': 28}), (True, {}), (True, {})]), \
             patch.object(queued, 'run_independent_audits', side_effect=lambda *a: order.append('audit')), \
             patch.object(queued, 'gpu_snapshot', side_effect=gpu), \
             patch.object(queued, 'run_evaluation', side_effect=evaluation):
            queued.main(self.args())
        self.assertEqual(order, ['audit', 'busy', 'idle', 'idle', 'smoke', 'idle', 'idle', 'full'])
        self.assertGreaterEqual(sleep.call_count, 3)
        status = queued.read_json(self.output/'queue_status.json')
        self.assertEqual(status['status'], 'complete')
        self.assertEqual((self.directory/'latest_run.txt').read_text(encoding='utf-8').strip(), str((self.output/'full').resolve()))

    def test_smoke_failure_prevents_full_evaluation(self):
        calls = []
        def fail_smoke(root, output, kit, disk, smoke): calls.append(smoke); raise RuntimeError('smoke failed')
        with self.main_patches(), patch.object(queued.time, 'sleep'), \
             patch.object(queued, 'rendering_complete', return_value=(True, {})), \
             patch.object(queued, 'run_independent_audits'), patch.object(queued, 'gpu_snapshot', return_value=self.idle), \
             patch.object(queued, 'run_evaluation', side_effect=fail_smoke):
            with self.assertRaisesRegex(RuntimeError, 'smoke failed'): queued.main(self.args())
        self.assertEqual(calls, [True])
        self.assertEqual(queued.read_json(self.output/'queue_status.json')['status'], 'failed')

    def test_failed_render_prevents_audits_and_evaluation(self):
        queued.write_json(self.state/'status.json', {'status': 'failed', 'error': 'worker failed'})
        with self.main_patches(), patch.object(queued, 'run_independent_audits') as audit, \
             patch.object(queued, 'run_evaluation') as evaluate:
            with self.assertRaisesRegex(RuntimeError, 'Rendering did not finish'): queued.main(self.args())
        audit.assert_not_called(); evaluate.assert_not_called()

    def test_successful_process_exit_with_partial_jobs_is_rejected(self):
        output = self.output/'smoke'
        def incomplete(command, log, environment=None):
            queued.write_json(output/'run_manifest.json', {'status': 'complete', 'expected_jobs': 15, 'completed_jobs': ['one']})
        with patch.object(queued, 'run_logged', side_effect=incomplete):
            with self.assertRaisesRegex(RuntimeError, 'expected jobs'):
                queued.run_evaluation(self.root, output, self.directory/'kit', self.directory/'disk.onnx', True)


if __name__ == '__main__': unittest.main()
