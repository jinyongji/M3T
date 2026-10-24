"""Independent paired RGB experiments; preserve old five-method runs."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from rgb_build import HERE,modules,verified
from rgb_report import analyze,regression,summarize

THREADS=['OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OPENCV_FOR_THREADS_NUM','LIBGL_ALWAYS_SOFTWARE','LP_NUM_THREADS']

def select(index,stage):
    configs=index['configurations']
    if stage=='smoke':
        wanted={(1,75,0),(6,45,3),(20,15,0)}
        result=[c for c in configs if (c['scene_id'],c['elevation'],c['gt_index']) in wanted]
        if len(result)!=3:raise ValueError('Unexpected smoke scope')
        return result,21
    if stage=='pilot':
        result=[c for c in configs if c['scene_id']<=5 and c['elevation'] in [15,45,75]]
        if len(result)!=63:raise ValueError('Expected 63 fixed pilot target rings')
        return result,72
    if stage=='full':
        if len(configs)!=966:raise ValueError('Expected all 966 target rings')
        return configs,72
    raise ValueError('Unknown stage')

def native(command,folder,data):
    folder.mkdir(parents=True,exist_ok=True);data.save(folder/'command.json',command)
    with (folder/'evaluation.log').open('w',encoding='utf-8') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        tail=(folder/'evaluation.log').read_text(encoding='utf-8',errors='replace').splitlines()[-35:]
        raise RuntimeError('Native RGB exit %d: %s\n%s'%(r.returncode,folder/'evaluation.log','\n'.join(tail)))

def execute(a):
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);path=a.output/'run_manifest.json'
    previous=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None
    if bool(previous)!=a.resume:raise ValueError('Choose new output or resume')
    if previous:
        for k,v in previous['identity']['launch'].items():setattr(a,k,Path(v) if k in ['dataset','cache','kit','tless_kit','disk_model','smoke_run'] and v else v)
    old,data=modules(a.tless_kit)
    import tracking_report as base
    import tracking_run as runner
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('Select exactly one GPU UUID')
    if data.sha(a.disk_model)!='5ff1ecfd9a31009a12d0e73ec76b476291756fe101b0417e3d0cc90a0dd42648':raise ValueError('Unexpected DISK weights')
    with contextlib.ExitStack() as stack:
        runner.lock(stack,HERE/'.build.lock',True);runner.lock(stack,a.output/'.run.lock')
        # Protect the shared immutable-source template cache from old/new T-LESS jobs.
        runner.lock(stack,a.cache/'.cache.lock');runner.lock(stack,a.tless_kit/'.build.lock',True)
        prepared,index=data.prepare(a.dataset,a.cache/'prepared',data.load(a.tless_kit/'expected_sources.json'))
        configs,frames=select(index,a.stage);inputs=data.rgb_preflight(a.dataset,configs,frames)
        build=verified(a.kit,a.tless_kit);reference=old.verified(a.kit,'ours')
        variants=data.load(HERE/'variants.json')
        if a.stage=='full':variants=[v for v in variants if v['id'] in ['B0','B3']]
        runtime=dict(gpu=gpu,threads={k:os.environ.get(k,'') for k in THREADS},python=sys.version,numpy=np.__version__)
        launch={k:str(getattr(a,k).resolve()) for k in ['dataset','cache','kit','tless_kit','disk_model']}
        launch.update(stage=a.stage,smoke_run=str(a.smoke_run.resolve()) if a.smoke_run else None)
        if a.stage!='smoke':
            if not a.smoke_run:raise ValueError('First pass the B0 vs frozen Ours regression smoke')
            sm=data.load(a.smoke_run/'run_manifest.json');gate=data.load(a.smoke_run/'regression_gate.json')
            smoke_configs,smoke_frames=select(index,'smoke');smoke_inputs=data.rgb_preflight(a.dataset,smoke_configs,smoke_frames)
            prior=sm['identity']
            if (not gate.get('passed') or sm['status']!='complete' or prior['launch']['stage']!='smoke' or prior['build']!=build
                    or prior['runtime']!=runtime or prior['metadata_fingerprint']!=index['fingerprint'] or prior['images']!=smoke_inputs):
                raise ValueError('RGB smoke regression failed or identity changed; inspect before pilot/full')
        identity=dict(launch=launch,metadata_fingerprint=index['fingerprint'],images=inputs,build=build,runtime=runtime,frames=frames,variants=variants,
            template_source=reference['binary_sha256'],protocol=data.PROTOCOL,
            smoke_gate_sha256=data.sha(a.smoke_run/'regression_gate.json') if a.smoke_run else None,
            scope='Exploratory RGB-only change; GT reset/metric/input protocol unchanged; CAD-DISK inactive')
        if previous and previous['identity']!=identity:raise ValueError('RGB resume identity changed')
        manifest=dict(status='running',pid=os.getpid(),identity=identity,expected_jobs=len(configs)*len(variants),expected_reference_jobs=len(configs) if a.stage=='smoke' else 0,
            completed_jobs=previous['completed_jobs'][:] if previous else [],output=str(a.output))
        data.save(path,manifest);done=set(manifest['completed_jobs']);records=[];comparisons=[];models={};scenes={}
        try:
            for j,c in enumerate(configs):
                order=variants[j%len(variants):]+variants[:j%len(variants)]
                # Model generation remains identical to the frozen Ours driver.
                cache=a.cache/'models'/index['fingerprint'][:16]/'ours'/reference['binary_sha256'][:16]/('obj_%06d'%c['obj_id'])
                common=['--dataset',str(a.dataset),'--prepared',str(prepared),'--id',c['id'],'--cache',str(cache),'--frames',str(frames),'--disk-model',str(a.disk_model)]
                prefix=['xvfb-run','-a','-s','-screen 0 1280x1024x24']
                for v in order:
                    key=c['id']+'/'+v['id'];folder=a.output/'configurations'/key/'ours';folder.mkdir(parents=True,exist_ok=True)
                    manifest.update(current=key,job=len(records)+1);data.save(path,manifest)
                    print('[%d/%d] %s'%(len(records)+1,manifest['expected_jobs'],key),flush=True)
                    if key not in done:native(prefix+[str(HERE/'build/evaluate_tless_rgb')]+common+['--output',str(folder),'--variant',v['id']],folder,data)
                    record,errors,times=analyze(folder,c,index,prepared,a.dataset,frames,v,models,scenes,data,base);records.append((record,errors,times))
                    if key not in done:done.add(key);manifest['completed_jobs'].append(key)
                    manifest['completed_evaluated_frames']=sum(r['frames'] for r,e,t in records);data.save(path,manifest)
                    print('  success %.4f%%; independent audit differences %d'%(100*record['successes']/record['frames'],len(record['audit_disagreements'])),flush=True)
                if a.stage=='smoke':
                    folder=a.output/'references'/c['id']/'ours';marker=folder/'complete.json'
                    if not marker.exists():native(prefix+[str(a.tless_kit/'build/ours/evaluate_tless')]+common+['--output',str(folder)],folder,data)
                    base.analyze(folder,c,index,prepared,a.dataset,frames,models,scenes);data.save(marker,dict(status='complete'))
                    result=regression(a.output/'configurations'/c['id']/'B0/ours',folder,base)
                    comparisons.append(dict(config_id=c['id'],**result));data.save(a.output/'regression.json',comparisons)
                    print('  B0 vs frozen Ours: '+('PASS' if result['passed'] else 'REVIEW REQUIRED'),flush=True)
            summarize(records,variants,a.output,a.stage,identity,data,base)
            if a.stage=='smoke':
                passed=len(comparisons)==len(configs) and all(r['passed'] for r in comparisons)
                data.save(a.output/'regression_gate.json',dict(passed=passed,comparisons=comparisons))
                if not passed:raise ValueError('B0 differs from frozen Ours; pilot/full blocked')
            manifest.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());data.save(path,manifest)
        except BaseException as e:
            manifest.update(status='failed',error=str(e));data.save(path,manifest);raise
    print('Complete: '+str(a.output),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['smoke','pilot','full'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke-run',type=Path)
    p.add_argument('--dataset',type=Path,default=Path('/home/jyj/autodl-tmp/bop_datasets/tless'))
    p.add_argument('--cache',type=Path,default=Path('/home/jyj/autodl-tmp/M3T-TLESS-cache'))
    p.add_argument('--kit',type=Path,default=Path('/home/jyj/Projects/industrial-tracker-comparison'))
    p.add_argument('--tless-kit',type=Path,default=Path('/home/jyj/Projects/tless-tracker-comparison'))
    p.add_argument('--disk-model',type=Path,default=Path('/home/jyj/Projects/M3T/disk_512.onnx'));execute(p.parse_args())
