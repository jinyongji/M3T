"""Wait for complete rendering, audit inputs, then run the five-method smoke/full evaluations."""
import argparse
import contextlib
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

KIT = Path(__file__).resolve().parent
CLEAN_KEYS = ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONHOME', 'PYTHONPATH',
              'LIBGL_ALWAYS_SOFTWARE', 'LP_NUM_THREADS', 'OMP_NUM_THREADS', 'OMP_DYNAMIC')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def clean_environment():
    result = {k: v for k, v in os.environ.items() if k not in CLEAN_KEYS}
    result.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return result


def controller_alive(pid):
    if int(pid) <= 0: return False
    try:
        # /proc is read-only here; avoid platform-specific signal semantics.
        status = Path('/proc/%d/stat' % int(pid)).read_text(encoding='utf-8')
        return status.rsplit(')', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError): return False
    except PermissionError: return True


def rendering_complete(state):
    status_file = state/'status.json'
    if not status_file.exists(): return False, {}
    status = read_json(status_file)
    if status.get('status') in ('failed', 'interrupted'):
        raise RuntimeError('Rendering did not finish: '+json.dumps(status, ensure_ascii=False))
    if status.get('status') != 'complete':
        if status.get('status') in ('rendering', 'preflight', 'retrying') and status.get('pid'):
            if not controller_alive(status['pid']):
                raise RuntimeError('Render controller exited before completion; resume rendering first')
        return False, status
    audit_file = state/'validation/final_audit.json'
    if not audit_file.exists(): raise RuntimeError('Render status is complete but final_audit.json is missing')
    audit = read_json(audit_file)
    if (audit.get('status'), audit.get('scope'), audit.get('images')) != ('passed', 'complete_dataset', 54000):
        raise RuntimeError('Complete 54,000-image rendering audit did not pass')
    return True, status


def gpu_snapshot(device):
    result = subprocess.run(['nvidia-smi', '-i', device,
                             '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                            check=True, text=True, stdout=subprocess.PIPE, env=clean_environment())
    records = list(csv.reader(result.stdout.strip().splitlines()))
    if len(records) != 1 or len(records[0]) != 3: raise RuntimeError('Select exactly one GPU')
    uuid, memory, utilization = [x.strip() for x in records[0]]
    applications = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                                 check=True, text=True, stdout=subprocess.PIPE, env=clean_environment())
    pids = [row[1].strip() for row in csv.reader(applications.stdout.splitlines())
            if len(row) >= 2 and row[0].strip() == uuid]
    return {'uuid': uuid, 'memory_used_mib': float(memory), 'utilization_percent': float(utilization), 'compute_pids': pids}


def gpu_idle(snapshot):
    return not snapshot['compute_pids'] and snapshot['memory_used_mib'] < 512 and snapshot['utilization_percent'] <= 10


def run_logged(command, log, environment=None):
    with Path(log).open('a', encoding='utf-8') as stream:
        subprocess.run([str(x) for x in command], check=True, stdout=stream, stderr=subprocess.STDOUT, env=environment)


def run_independent_audits(root, output):
    blender = root/'中间结果文件/runtime_advanced_v1/extracted/blender-4.5.4-linux-x64/blender'
    if not blender.is_file(): raise RuntimeError('Packaged Blender is missing: '+str(blender))
    audit = root/'中间结果文件/run_state_advanced_v1/independent_audit'; audit.mkdir(parents=True, exist_ok=True)
    for script, name in (('audit_inputs.py', 'inputs.json'), ('audit_scenes.py', 'scenes.json')):
        # Pure geometry/pose checks; preserve the renderer's complete status.
        run_logged([blender, '--background', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
                    '--python', KIT/script, '--', '--root', root, '--output', audit/name],
                   output/(name+'.log'), clean_environment())
    inputs = read_json(audit/'inputs.json'); scenes = read_json(audit/'scenes.json')
    index = read_json(audit/'evaluation_index.json')
    if inputs.get('status') != 'passed' or inputs.get('pose_files') != 324 or inputs.get('sequences') != 108:
        raise RuntimeError('Independent input audit is incomplete')
    if (scenes.get('status'), scenes.get('scope'), scenes.get('sequences_checked'), scenes.get('total_frame_checks')) != (
            'passed', 'all integer frames', 108, 54000):
        raise RuntimeError('Independent Blender audit is incomplete')
    if len(index['configurations']) != 324 or sum(bool(r['is_focus']) for r in index['configurations']) != 108:
        raise RuntimeError('Expected 324 target configurations, including 108 focus configurations')


def run_evaluation(root, output, comparison_kit, disk_model, smoke):
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-u', KIT/'evaluate.py', '--root', root, '--output', output,
               '--kit', comparison_kit, '--disk-model', disk_model, '--frames', '140' if smoke else '500']
    if smoke: command.append('--smoke')
    run_logged(command, output/'launcher.log')
    manifest = read_json(output/'run_manifest.json')
    expected = 15 if smoke else 1620
    if manifest.get('status') != 'complete' or manifest.get('expected_jobs') != expected or len(manifest.get('completed_jobs', [])) != expected:
        raise RuntimeError('Evaluation did not complete all expected jobs: '+str(output))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kit', type=Path, default=Path('/home/jyj/Projects/industrial-tracker-comparison'))
    p.add_argument('--disk-model', type=Path, default=Path('/home/jyj/Projects/M3T/disk_512.onnx'))
    p.add_argument('--poll-seconds', type=float, default=30)
    args = p.parse_args(argv)
    if not 1 <= args.poll_seconds <= 300: p.error('poll-seconds must be 1..300')
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not device or ',' in device: p.error('Select exactly one allocated GPU in CUDA_VISIBLE_DEVICES')
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = root/'中间结果文件/run_state_advanced_v1'; state.mkdir(parents=True, exist_ok=True)
    info = {'pid': os.getpid(), 'root': str(root), 'output': str(output), 'selected_device': device,
            'smoke_output': str(output/'smoke'), 'full_output': str(output/'full')}
    def update(stage, **details):
        previous = info.get('status'); info.update(status=stage, **details)
        info['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(output/'queue_status.json', info)
        if stage != previous: print(stage+': '+str(output), flush=True)
    def wait_for_gpu(next_stage):
        idle_polls = 0
        while idle_polls < 2:
            snapshot = gpu_snapshot(device)
            idle_polls = idle_polls+1 if gpu_idle(snapshot) else 0
            update('waiting_gpu', next_stage=next_stage, gpu=snapshot, consecutive_idle_polls=idle_polls)
            if idle_polls < 2: time.sleep(args.poll_seconds)
        os.environ['CUDA_VISIBLE_DEVICES'] = snapshot['uuid']
    with contextlib.ExitStack() as resources:
        import fcntl
        lock = resources.enter_context((state/'evaluation_queue.lock').open('a'))
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another after-render evaluation queue is active for this dataset')
        (output.parent/'latest_queue.txt').write_text(str(output)+'\n', encoding='utf-8')
        try:
            while True:
                ready, progress = rendering_complete(state)
                update('waiting_render', rendering=progress)
                if ready:
                    render_lock = (state/'render.lock').open('a')
                    try: fcntl.flock(render_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError: render_lock.close()
                    else:
                        resources.enter_context(render_lock)
                        ready, _ = rendering_complete(state)
                        if not ready: raise RuntimeError('Rendering state changed during handoff; rerun the queue')
                        break
                time.sleep(args.poll_seconds)
            update('auditing_inputs')
            run_independent_audits(root, output)
            wait_for_gpu('smoke')
            update('smoke_running')
            (output.parent/'latest_run.txt').write_text(str(output/'smoke')+'\n', encoding='utf-8')
            run_evaluation(root, output/'smoke', args.kit, args.disk_model, True)
            wait_for_gpu('full')
            update('full_running')
            (output.parent/'latest_run.txt').write_text(str(output/'full')+'\n', encoding='utf-8')
            run_evaluation(root, output/'full', args.kit, args.disk_model, False)
            update('complete')
        except BaseException as error:
            update('failed', error=str(error)); (output/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
            raise


if __name__ == '__main__': main()
