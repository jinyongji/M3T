#!/usr/bin/env python3
"""Sequential RBOT comparison using separately built drivers and frozen algorithms."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from rbot_data import (BODIES, SEQUENCES, METHODS, DISK_SHA256, configurations, preflight,
                       sha256, write_json, write_normalized_poses, old_run_assessment)
from rbot_build import HERE, verified
from rbot_metrics import analyze
from rbot_report import summarize


def hold(resources, path, shared=False):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = resources.enter_context(path.open('a'))
    try:
        fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another benchmark/build holds '+str(path))


def jobs(configs, methods):
    for i, config in enumerate(configs):
        offset = i % len(methods)
        for method in methods[offset:]+methods[:offset]:
            yield config, method


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('/home/jyj/autodl-tmp/RBOT-dataset'))
    p.add_argument('--kit', type=Path, default=Path(os.environ.get('M3T_COMPARISON_KIT', '/home/jyj/Projects/industrial-tracker-comparison')))
    p.add_argument('--disk-model', type=Path, default=Path('/home/jyj/Projects/M3T/disk_512.onnx'))
    p.add_argument('--cache', type=Path, default=Path('/home/jyj/autodl-tmp/M3T-RBOT-comparison-cache'))
    p.add_argument('--prior-run', type=Path, default=Path('/home/jyj/autodl-tmp/M3T-rbot-runs/linux_final_all_first4_20261015T091045Z_e41869dc'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--body', action='append', choices=BODIES)
    p.add_argument('--sequence', action='append', choices=SEQUENCES)
    p.add_argument('--frames', type=int, default=1001, help='Includes GT initialization image 0000')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if args.resume:
        previous = json.loads((args.output/'run_manifest.json').read_text(encoding='utf-8'))
        for key, value in previous['identity']['launch'].items():
            setattr(args, key, Path(value) if key in ['dataset', 'kit', 'disk_model', 'cache', 'prior_run'] else value)
    elif args.smoke:
        args.frames = 21
        args.body = ['ape', 'driller']
        args.sequence = SEQUENCES
    if len(set(args.methods)) != len(args.methods) or not 2 <= args.frames <= 1001 or not 0 <= args.warmup < args.frames-1:
        p.error('Unique methods; frames 2..1001; warmup must leave timed frames')
    for key in ['body', 'sequence']:
        values = getattr(args, key)
        if values and len(values) != len(set(values)):
            p.error('Duplicate '+key)
    return args


def execute(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    root, kit, cache = args.dataset.resolve(), args.kit.resolve(), args.cache.resolve()
    configs = configurations(args.body, args.sequence)
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('Select exactly one allocated GPU UUID before running')
    if not shutil.which('xvfb-run'):
        raise ValueError('Source the existing rootless dependency environment first')
    if 'ours' in args.methods and sha256(args.disk_model) != DISK_SHA256:
        raise ValueError('Use the previously verified disk_512.onnx model')
    with contextlib.ExitStack() as resources:
        hold(resources, HERE/'.build.lock', shared=True)
        hold(resources, output/'.run.lock')
        hold(resources, cache/'.cache.lock')
        hold(resources, Path('/home/jyj/autodl-tmp/M3T-industrial-comparison-cache/.cache.lock'))
        print('Checking RBOT calibration, 1001 poses, model assets and all selected PNG headers...', flush=True)
        inputs, truth = preflight(root, configs, args.frames)
        builds = {m: verified(kit, m) for m in args.methods}
        normalized = cache/'inputs'/inputs['pose_sha256'][:16]/'poses_first_m.txt'
        write_normalized_poses(normalized, truth)
        launch = {k: str(getattr(args, k).resolve()) for k in ['dataset', 'kit', 'disk_model', 'cache', 'prior_run']}
        launch.update(methods=args.methods, body=args.body, sequence=args.sequence, frames=args.frames,
                      warmup=args.warmup, smoke=args.smoke)
        identity = dict(dataset=str(root), methods=args.methods, configurations=configs, frames=args.frames,
                        warmup=args.warmup, repeats=1, launch=launch, inputs=inputs, builds=builds,
                        primary_timing_metric='compute_wall_us', selected_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                        threads={k: os.environ.get(k, '') for k in ['OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OPENCV_FOR_THREADS_NUM',
                                                                  'LIBGL_ALWAYS_SOFTWARE', 'LP_NUM_THREADS']},
                        protocol='RGB; initialize 0000; evaluate 0001..N; 5cm/5deg; invalid=failure; count failure then GT reset; no modeled occluder')
        path = output/'run_manifest.json'
        old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        if bool(old) != args.resume:
            raise ValueError('Choose a new output or explicitly resume the existing run')
        if old and old['identity'] != identity:
            raise ValueError('Inputs, algorithm/build, GPU, thread policy or adapter changed; start a new run')
        manifest = dict(status='running', pid=os.getpid(), output=str(output), identity=identity,
                        expected_jobs=len(configs)*len(args.methods), completed_jobs=list(old['completed_jobs']) if old else [],
                        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        write_json(path, manifest)
        legacy = old_run_assessment(args.prior_run)
        write_json(output/'previous_ours_run.json', legacy)
        done = set(manifest['completed_jobs']); results = []
        try:
            for config, method in jobs(configs, args.methods):
                key = config['id']+'/'+method; directory = output/'configurations'/key
                directory.mkdir(parents=True, exist_ok=True)
                manifest.update(current=dict(config, method=method), job=len(results)+1)
                write_json(path, manifest)
                if key not in done:
                    model_key = inputs.get('model_fingerprints', {}).get(config['body'], inputs['model_fingerprint'])
                    model_cache = cache/'models'/model_key[:16]/method/builds[method]['binary_sha256'][:16]
                    model_cache.mkdir(parents=True, exist_ok=True)
                    cmd = ['xvfb-run', '-a', '-s', '-screen 0 1280x1024x24', str(HERE/'build'/method/'evaluate_rbot_comparison'),
                           '--dataset', str(root), '--cache', str(model_cache), '--output', str(directory),
                           '--body', config['body'], '--sequence', config['sequence'], '--frames', str(args.frames), '--poses', str(normalized)]
                    if method == 'ours':
                        cmd += ['--disk-model', str(args.disk_model.resolve())]
                    write_json(directory/'command.json', cmd)
                    print('[%d/%d] %s' % (len(results)+1, manifest['expected_jobs'], key), flush=True)
                    log_file = directory/'evaluation.log'
                    try:
                        with log_file.open('a', encoding='utf-8') as log:
                            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as error:
                        tail = '\n'.join(log_file.read_text(encoding='utf-8', errors='replace').splitlines()[-35:])
                        raise RuntimeError('Native tracker exit %d; log: %s\n%s' % (error.returncode, log_file, tail)) from error
                item = analyze(directory, config, truth, args.frames)
                item.update(method=method, repeat=1)
                write_json(directory/'metrics.json', item)
                results.append(item)
                if key not in manifest['completed_jobs']:
                    manifest['completed_jobs'].append(key)
                write_json(path, manifest)
                print('  success %.4f%%; audit %.4f%%; invalid %d' %
                      (item['success_percent'], item['recomputed_success_percent'], item['invalid_estimate_count']), flush=True)
            summarize(output, results, identity, legacy)
            manifest.update(status='complete'); write_json(path, manifest)
            print('Complete: '+str(output), flush=True)
        except BaseException as error:
            manifest.update(status='failed', error=str(error)); write_json(path, manifest)
            raise


def status(output):
    path = Path(output)/'run_manifest.json'
    if path.is_file():
        data = json.loads(path.read_text(encoding='utf-8'))
        compact = {k: data[k] for k in ['status', 'pid', 'output', 'job', 'expected_jobs', 'current', 'error'] if k in data}
        compact['completed_jobs'] = len(data['completed_jobs'])
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    log = Path(output)/'launcher.log'
    if log.is_file():
        print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-15:]))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'status':
        status(sys.argv[2])
    else:
        execute(parse_args())
