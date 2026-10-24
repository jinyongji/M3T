"""T-LESS metadata preparation and sequential five-method execution/resume."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tracking_data import METHODS,PROTOCOL,load,prepare,rgb_preflight,save,sha
from tracking_build import HERE,verified
from tracking_report import analyze,summarize

DEFAULT='/home/jyj/autodl-tmp/bop_datasets/tless'
CACHE='/home/jyj/autodl-tmp/M3T-TLESS-cache'


def lock(stack,path,shared=False):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(path.open('a'))
    try:fcntl.flock(f,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Another task holds '+str(path))


def metadata_prepare(root,cache):
    result,index=prepare(root,Path(cache)/'prepared',load(HERE/'expected_sources.json'))
    print('Prepared: '+str(result),flush=True)
    print(json.dumps(dict(index['metadata_audit'],normalized_size=index['image_size'],
                          evaluated_frames_per_method=index['evaluated_frames_per_method'],
                          timed_frames_per_method=index['timed_frames_per_method'])),flush=True)
    return result,index


def execute(a):
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output/'run_manifest.json'
    previous=load(manifest_path) if manifest_path.exists() else None
    if bool(previous)!=a.resume:raise ValueError('Choose a new output or use resume')
    if previous:
        for k,v in previous['identity']['launch'].items():setattr(a,k,Path(v) if k in ['dataset','cache','kit','disk_model'] else v)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('Select exactly one GPU UUID')
    if not shutil.which('xvfb-run'):raise ValueError('Source existing dependency environments first')
    if sha(a.disk_model)!='5ff1ecfd9a31009a12d0e73ec76b476291756fe101b0417e3d0cc90a0dd42648':raise ValueError('DISK model differs from the verified full configuration')
    with contextlib.ExitStack() as stack:
        lock(stack,HERE/'.build.lock',True);lock(stack,a.cache/'.cache.lock');lock(stack,a.output/'.run.lock')
        prepared,index=metadata_prepare(a.dataset,a.cache)
        if a.smoke:
            wanted=[(1,75,0),(6,45,3),(20,15,0)]
            configs=[c for c in index['configurations'] if (c['scene_id'],c['elevation'],c['gt_index']) in wanted];frames=21
        else:configs=index['configurations'];frames=72
        inputs=rgb_preflight(a.dataset,configs,frames);builds={m:verified(a.kit,m) for m in METHODS}
        launch={k:str(getattr(a,k).resolve()) for k in ['dataset','cache','kit','disk_model']};launch['smoke']=a.smoke
        identity=dict(launch=launch,metadata_fingerprint=index['fingerprint'],index_sha256=sha(prepared/'index.json'),
                      images=inputs,builds=builds,selected_gpu=gpu,frames=frames,
                      threads={k:os.environ.get(k,'') for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','OPENCV_FOR_THREADS_NUM','LIBGL_ALWAYS_SOFTWARE','LP_NUM_THREADS']})
        if previous and previous['identity']!=identity:raise ValueError('Code, inputs, GPU or timing policy changed; start a new run')
        manifest=dict(status='running',pid=os.getpid(),output=str(a.output),identity=identity,
                      expected_jobs=len(configs)*5,completed_jobs=list(previous['completed_jobs']) if previous else [])
        save(manifest_path,manifest);done=set(manifest['completed_jobs']);results=[];models={};scenes={}
        try:
            for j,c in enumerate(configs):
                methods=METHODS[j%5:]+METHODS[:j%5]
                for method in methods:
                    key=c['id']+'/'+method;folder=a.output/'configurations'/key;folder.mkdir(parents=True,exist_ok=True)
                    manifest.update(current=key,job=len(results)+1);save(manifest_path,manifest)
                    print('[%d/%d] %s'%(len(results)+1,manifest['expected_jobs'],key),flush=True)
                    if key not in done:
                        cache=a.cache/'models'/index['fingerprint'][:16]/method/builds[method]['binary_sha256'][:16]/('obj_%06d'%c['obj_id'])
                        command=['xvfb-run','-a','-s','-screen 0 1280x1024x24',str(HERE/'build'/method/'evaluate_tless'),
                                 '--dataset',str(a.dataset),'--prepared',str(prepared),'--id',c['id'],'--output',str(folder),
                                 '--cache',str(cache),'--frames',str(frames),'--disk-model',str(a.disk_model)]
                        save(folder/'command.json',command)
                        with (folder/'evaluation.log').open('w',encoding='utf-8') as log:
                            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                        if run.returncode:
                            tail=(folder/'evaluation.log').read_text(encoding='utf-8',errors='replace').splitlines()[-35:]
                            raise RuntimeError('Native evaluation failed (%d): %s\n%s'%(run.returncode,folder/'evaluation.log','\n'.join(tail)))
                    record,errors,timing=analyze(folder,c,index,prepared,a.dataset,frames,models,scenes);record['method']=method
                    results.append((record,errors,timing));print('  success %.4f%%; independent audit disagreements %d'%(100*record['successes']/record['frames'],len(record['audit_disagreements'])),flush=True)
                    if key not in done:done.add(key);manifest['completed_jobs'].append(key)
                    manifest['completed_evaluated_frames']=sum(r[0]['frames'] for r in results);save(manifest_path,manifest)
            summarize(results,index,a.output,a.smoke);manifest.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(manifest_path,manifest)
        except BaseException as e:
            manifest.update(status='failed',error=str(e));save(manifest_path,manifest);raise
    print('Complete: '+str(a.output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--dataset',type=Path,default=Path(DEFAULT));p.add_argument('--cache',type=Path,default=Path(CACHE))
    p.add_argument('--kit',type=Path,default=Path('/home/jyj/Projects/industrial-tracker-comparison'))
    p.add_argument('--disk-model',type=Path,default=Path('/home/jyj/Projects/M3T/disk_512.onnx'))
    p.add_argument('--output',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':
        with contextlib.ExitStack() as stack:
            lock(stack,a.cache/'.cache.lock');metadata_prepare(a.dataset,a.cache)
    elif a.output is None:p.error('--output is required for run')
    else:execute(a)
