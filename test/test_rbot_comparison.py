"""RBOT frame alignment, resets, timing scope and runner recovery contracts."""
import contextlib
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/rbot_compare'))
import rbot_data as data
import rbot_metrics as metrics
import rbot_report as report
import compare_rbot as runner
from generate_adapters import generate


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = data.configurations(['ape'], ['a_regular'])[0]
        self.truth = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, .5] for _ in range(1001)]
        self.csv = self.root/'native/results/poses/poses_a_regular_ape.csv'
        self.csv.parent.mkdir(parents=True)
        self.rows = []
        for i in range(2):
            row = dict(frame=i, image_index=i+1, image_file='a_regular%04d.png' % (i+1), segment_id=0,
                       estimate_source='single', pose_finite=1, reset_primary_after_frame=0,
                       reset_secondary_after_frame=0, tracking_success=1, translation_error_m=0, rotation_error_rad=0)
            for prefix in ['est', 'gt']:
                for r in range(3):
                    for c in range(3):
                        row['%s_r%d%d' % (prefix, r, c)] = int(r == c)
                for axis, value in zip('xyz', [0, 0, .5]):
                    row['%s_t%s_m' % (prefix, axis)] = value
            row.update({k: 0 for k in metrics.TIMES})
            row.update(cycle_us=1000, decode_us=100000, preprocessing_us=10, tracking_wall_us=101000,
                       compute_wall_us=1000, metric_us=20, reset_us=0, frame_wall_us=101020,
                       correspondences_us=700, optimization_us=300)
            self.rows.append(row)
        self.save()
        self.meta = dict(schema_version=1, object='ape', sequence='a_regular', translation_unit='m',
                         model_unit_in_meter=.001, first_image_index=1, last_image_index=2, frame_count=2,
                         estimate_capture='final_selected_output_before_gt_reset', intrinsics=data.INTRINSICS,
                         translation_threshold_m=.05, rotation_threshold_rad=math.radians(5))
        data.write_json(self.csv.with_name('metadata_a_regular_ape.json'), self.meta)

    def save(self):
        with self.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0])); writer.writeheader(); writer.writerows(self.rows)

    def analyze(self):
        return metrics.analyze(self.root/'native', self.config, self.truth, 3)


class DataTests(Fixture):
    def test_full_scope_is_72_configurations_and_360_jobs(self):
        configs = data.configurations()
        self.assertEqual(len(configs), 72)
        self.assertEqual(len({c['id'] for c in configs}), 72)
        jobs = list(runner.jobs(configs, data.METHODS))
        self.assertEqual(len(jobs), 360)
        self.assertEqual(jobs[0][1], 'rbgt')
        self.assertEqual(jobs[5][1], 'srt3d')
        self.assertEqual(72*1000, 72000)
        self.assertEqual(72*990, 71280)

    def test_units_and_1001_pose_rows(self):
        path = self.root/'poses_first.txt'
        path.write_text('r11 r12 r13 ...\n'+('1 0 0 0 1 0 0 0 1 0 0 500\n'*1001))
        poses = data.read_poses(path)
        self.assertEqual(poses, self.truth)
        data.write_normalized_poses(self.root/'clean.txt', poses)
        self.assertEqual(float((self.root/'clean.txt').read_text().splitlines()[0].split()[-1]), .5)
        path.write_text(path.read_text().splitlines()[0]+'\n'+'1 0 0 0 1 0 0 0 1 0 0 500\n')
        with self.assertRaisesRegex(ValueError, '1001'):
            data.read_poses(path)

    def test_old_cycle_only_results_cannot_supply_new_timing(self):
        path = self.root/'legacy/results/poses'; path.mkdir(parents=True)
        (path/'poses_test.csv').write_text('frame,cycle_us\n0,1000\n')
        assessment = data.old_run_assessment(path.parent.parent)
        self.assertTrue(assessment['rerun_ours_for_paired_timing'])
        self.assertFalse(assessment['has_comparable_timing_columns'])

    def test_only_referenced_model_assets_affect_input_identity(self):
        root = self.root/'data'; directory = root/'ape'; frames = directory/'frames'
        frames.mkdir(parents=True)
        (root/'poses_first.txt').write_text('header\n'+'1 0 0 0 1 0 0 0 1 0 0 500\n'*1001)
        (directory/'ape.obj').write_text('mtllib ape.obj.mtl\nv 0 0 0\n')
        (directory/'ape.obj.mtl').write_text('newmtl m\nmap_Kd ape_tex.png\n')
        (directory/'ape_tex.png').write_bytes(b'material bytes')
        header = b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',640,512)+bytes([8,2,0,0,0])+bytes(4)
        for i in range(3):
            (frames/('a_regular%04d.png' % i)).write_bytes(header)
        before, _ = data.preflight(root, [self.config], 3)
        generated = directory/'output'; generated.mkdir()
        (generated/'diagnostic.png').write_bytes(b'unrelated output')
        (directory/'ape_simple.obj').write_text('unrelated geometry')
        after, _ = data.preflight(root, [self.config], 3)
        self.assertEqual(before, after)
        self.assertEqual(set(before['asset_sha256']), {'ape/ape.obj', 'ape/ape.obj.mtl', 'ape/ape_tex.png'})
        (directory/'ape_tex.png').write_bytes(b'changed material')
        changed, _ = data.preflight(root, [self.config], 3)
        self.assertNotEqual(before['input_fingerprint'], changed['input_fingerprint'])


class MetricTests(Fixture):
    def test_image_zero_initialization_then_evaluate_one(self):
        self.meta['model_unit_in_meter'] = .0010000000474974513
        data.write_json(self.csv.with_name('metadata_a_regular_ape.json'), self.meta)
        item = self.analyze()
        self.assertEqual(item['frames'], 2)
        self.assertEqual(item['success_percent'], 100)
        self.rows[0]['image_index'] = 2; self.save()
        with self.assertRaisesRegex(ValueError, 'alignment'):
            self.analyze()

    def test_failure_counted_before_reset(self):
        self.rows[0].update(est_tx_m=.051, translation_error_m=.051, tracking_success=0,
                            reset_primary_after_frame=1)
        self.rows[1]['segment_id'] = 1; self.save()
        item = self.analyze()
        self.assertEqual(item['success_percent'], 50)
        self.assertEqual(item['reset_count'], 1)
        self.rows[1]['segment_id'] = 0; self.save()
        with self.assertRaisesRegex(ValueError, 'alignment'):
            self.analyze()

    def test_nonfinite_estimate_is_failure_not_dropped(self):
        self.rows[0].update(est_tx_m=float('nan'), pose_finite=0, tracking_success=0, reset_primary_after_frame=1,
                            translation_error_m=float('inf'), rotation_error_rad=float('inf'))
        self.rows[1]['segment_id'] = 1; self.save()
        item = self.analyze()
        self.assertEqual(item['frames'], 2)
        self.assertEqual(item['invalid_estimate_count'], 1)
        self.assertEqual(item['success_percent'], 50)

    def test_wrong_gt_or_calibration_is_rejected(self):
        self.rows[0]['gt_tz_m'] = .6; self.save()
        with self.assertRaisesRegex(ValueError, 'GT differs'):
            self.analyze()
        self.rows[0]['gt_tz_m'] = .5; self.save()
        self.meta['intrinsics'] = dict(data.INTRINSICS, cx=324.328)
        data.write_json(self.csv.with_name('metadata_a_regular_ape.json'), self.meta)
        with self.assertRaisesRegex(ValueError, 'intrinsics'):
            self.analyze()

    def test_compute_timing_excludes_large_decode_and_warmup_only_from_time(self):
        item = self.analyze()
        items = [dict(item, method=m, repeat=1) for m in ['rbgt', 'm3t_original_rgb', 'ours']]
        identity = dict(methods=['rbgt', 'm3t_original_rgb', 'ours'], configurations=[self.config], warmup=1)
        summary = report.summarize(self.root, items, identity)
        m = summary['methods']['ours']
        self.assertEqual(m['frames'], 2)
        self.assertEqual(m['timing']['compute_wall_us']['frames'], 1)
        self.assertEqual(m['timing']['compute_wall_us']['mean_us'], 1000)
        self.assertEqual(m['derived_compute_fps'], 1000)
        self.assertEqual(summary['primary_timing_metric'], 'compute_wall_us')
        text = (self.root/'results.md').read_text(encoding='utf-8')
        self.assertIn('计算均值', text)
        self.assertNotIn('完整帧均值', text)

    def test_corrupt_timing_or_old_schema_rejected(self):
        self.rows[0]['compute_wall_us'] = 2000; self.save()
        with self.assertRaisesRegex(ValueError, 'decomposition'):
            metrics.timing_rows(self.csv, 0)
        self.csv.write_text('frame,cycle_us\n0,1000\n')
        with self.assertRaisesRegex(ValueError, 'rerun'):
            metrics.timing_rows(self.csv, 0)

    def test_missing_method_prevents_complete_report(self):
        identity = dict(methods=['rbgt', 'ours'], configurations=[self.config], warmup=0)
        with self.assertRaisesRegex(ValueError, 'Incomplete/unpaired'):
            report.summarize(self.root, [dict(self.analyze(), method='ours')], identity)


class DriverTests(Fixture):
    def test_generated_drivers_keep_algorithm_steps_and_rbot_units(self):
        target = self.root/'generated'
        generate(ROOT/'examples/industrial_compare/baseline.cpp', ROOT, target)
        baseline = (target/'rbot_baseline.cpp').read_text()
        ours = (target/'rbot_ours.cpp').read_text()
        self.assertIn('in.mesh,.001f,true,false', baseline)
        self.assertIn('index_=0', baseline)
        self.assertIn('image_.rows==512 && image_.cols==640', baseline)
        self.assertIn('input.pose_row_count=1001', ours)
        self.assertIn('input.first_image_index=0', ours)
        self.assertIn('set_use_directional_temporal_prior(true)', ours)
        self.assertIn('const bool benchmark_mode=true', ours)

    def test_runner_resume_reaudits_and_does_not_execute_done_jobs(self):
        args = runner.parse_args(['--output', str(self.root/'out'), '--dataset', str(self.root/'data'),
                                  '--cache', str(self.root/'cache'), '--body', 'ape', '--sequence', 'a_regular',
                                  '--methods', 'rbgt', 'srt3d', '--frames', '3', '--warmup', '1'])
        inputs = dict(input_fingerprint='input1', model_fingerprint='model1', pose_sha256='pose1')
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            directory = Path(command[command.index('--output')+1])
            shutil.copytree(self.root/'native/results', directory/'results', dirs_exist_ok=True)
        with patch.object(runner, 'hold'), patch.object(runner, 'preflight', return_value=(inputs, self.truth)), \
             patch.object(runner, 'verified', side_effect=lambda kit, m: {'binary_sha256': m+'hash'}), \
             patch.object(runner.shutil, 'which', return_value='xvfb-run'), \
             patch.object(runner.subprocess, 'run', side_effect=fake_run), \
             patch.dict(runner.os.environ, CUDA_VISIBLE_DEVICES='GPU-test'):
            runner.execute(args)
            self.assertEqual(len(calls), 2)
            args.resume = True; runner.execute(args)
            self.assertEqual(len(calls), 2)
            inputs['input_fingerprint'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.execute(args)


if __name__ == '__main__':
    unittest.main()
