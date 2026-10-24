"""One dataset per GPU; nine variants sequential, independent caches, audited resume."""
import argparse
import contextlib
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from ablation_data import HERE,LANES,load,prepare,protect,variants,write_job,write_json,sha256,DISK_SHA256
from ablation_build import verified,old
from ablation_audit import analyze,regression
from ablation_report import summarize

THREADS=['OMP_NUM_THREADS','OMP_DYNAMIC','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OPENCV_FOR_THREADS_NUM','LIBGL_ALWAYS_SOFTWARE','LP_NUM_THREADS']


def hold(stack,path,shared=False):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(path.open('a'))
    try:fcntl.flock(f,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Another task holds '+str(path))


def native(command,folder):
    folder.mkdir(parents=True,exist_ok=True);write_json(folder/'command.json',command)
    with (folder/'evaluation.log').open('w',encoding='utf-8') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        tail=(folder/'evaluation.log').read_text(encoding='utf-8',errors='replace').splitlines()[-40:]
        raise RuntimeError('Native exit %d: %s\n%s'%(r.returncode,folder/'evaluation.log','\n'.join(tail)))


def link(source,destination):
    source=Path(source).resolve();destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        if destination.resolve()!=source:raise ValueError('Changed reference view')
    elif destination.exists():raise ValueError('Unexpected reference view contents')
    else:destination.symlink_to(source,target_is_directory=source.is_dir())


def model_cache_key(build,prior=None):
    if prior is None:return build['binary_sha256'][:16]
    previous=prior['identity']['build']
    if previous['sources']!=build['sources']:raise ValueError('Cannot reuse templates from changed frozen engines')
    for name in ['build_inputs/rbot_evaluator.cpp','build_inputs/rbot_evaluator.h']:
        if previous['helpers'].get(name)!=build['helpers'].get(name):raise ValueError('Cannot reuse templates from a changed evaluator/model generator')
    key=prior['identity'].get('model_cache_key',previous['binary_sha256'][:16])
    if len(key)!=16 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid template cache identity')
    return key


def reference_command(a,job,method,folder,cache):
    config=job['config'];dataset=Path(job['dataset']);reference=dict(job)
    if a.lane=='rbot':
        command=[str(a.rbot_kit/'build'/method/'evaluate_rbot_comparison'),'--dataset',str(dataset),'--body',job['object'],
                 '--sequence',job['sequence'],'--poses',job['poses']]
    else:
        condition=config['condition']
        if a.lane=='industrial_advanced':
            view=folder/'input_view';view.mkdir(parents=True,exist_ok=True);condition='01_normal'
            for name in ['cameras','objects']:link(dataset/name,view/name)
            relative=Path(config['scene'])/config['motion']/condition
            link(job['images'],view/'images'/relative);link(Path(job['poses']).parent,view/'poses'/relative)
            dataset=view;reference['sequence']=config['scene']+'__'+config['motion']+'__'+condition
        command=[str(a.kit/'build'/method/'evaluate_industrial_baseline'),'--dataset',str(dataset),
                 '--scene',config['scene'],'--motion',config['motion'],'--condition',condition,'--object',job['object']]
        if method=='ours':command+=['--benchmark']
    command+=['--cache',str(cache),'--output',str(folder),'--frames',str(job['frames'])]
    if method=='ours':command+=['--disk-model',str(a.disk_model)]
    return ['xvfb-run','-a','-s','-screen 0 1280x1024x24']+command,reference


def run(a):
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);manifest_path=a.output/'run_manifest.json'
    previous=load(manifest_path) if manifest_path.exists() else None
    if bool(previous)!=a.resume:raise ValueError('Choose new output or resume the existing run')
    if previous:
        for key,value in previous['identity']['launch'].items():setattr(a,key,Path(value) if key in ['dataset','cache','kit','rbot_kit','disk_model','smoke_run','reuse_model_run'] and value else value)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('Select exactly one GPU UUID')
    if not shutil.which('xvfb-run'):raise ValueError('Source the existing dependency environment')
    if sha256(a.disk_model)!=DISK_SHA256:raise ValueError('Use the previously verified DISK weights')
    with contextlib.ExitStack() as stack:
        hold(stack,HERE/'.build.lock',True);hold(stack,a.output/'.run.lock');hold(stack,a.cache/'.cache.lock');protect(stack,a.lane,a.dataset,hold)
        print('Validating complete selected poses, model assets and PNG headers...',flush=True)
        jobs,inputs=prepare(a.lane,a.dataset,a.cache,a.smoke);build=verified(a.kit)
        refs={m:old(a.kit,m) for m in ['ours','m3t_original_rgb']}
        if a.lane=='rbot':
            sys.path.insert(0,str(a.rbot_kit))
            from rbot_build import verified as verified_rbot
            refs={m:verified_rbot(a.kit,m) for m in ['ours','m3t_original_rgb']}
        runtime=dict(gpu=gpu,threads={k:os.environ.get(k,'') for k in THREADS},python=sys.version,numpy=np.__version__)
        gate=None
        template_key=model_cache_key(build)
        if a.reuse_model_run:
            prior=load(a.reuse_model_run/'run_manifest.json')
            if prior['identity']['launch']['lane']!=a.lane or Path(prior['identity']['launch']['cache']).resolve()!=a.cache.resolve():
                raise ValueError('Template cache reuse requires the same dataset lane/cache root')
            template_key=model_cache_key(build,prior)
            print('Reusing unchanged Region model generator cache: '+template_key,flush=True)
        if not a.smoke:
            if not a.smoke_run:raise ValueError('Full requires --smoke-run with a passed regression smoke')
            gate=load(a.smoke_run/'regression_gate.json');sm=load(a.smoke_run/'run_manifest.json')
            _,current_smoke=prepare(a.lane,a.dataset,a.cache,True)
            if (not gate.get('passed') or sm.get('status')!='complete' or not sm['identity']['launch']['smoke']
                    or sm['identity']['launch']['lane']!=a.lane or sm['identity']['build']!=build or sm['identity']['runtime']!=runtime
                    or sm['identity']['inputs']!=current_smoke or sm['identity']['references']!=refs):
                raise ValueError('Regression smoke failed or changed; rerun smoke and inspect regression.json')
            template_key=sm['identity'].get('model_cache_key',build['binary_sha256'][:16])
        launch=dict(lane=a.lane,dataset=str(a.dataset.resolve()),cache=str(a.cache.resolve()),kit=str(a.kit.resolve()),
                    rbot_kit=str(a.rbot_kit.resolve()),disk_model=str(a.disk_model.resolve()),smoke=a.smoke,
                    smoke_run=str(a.smoke_run.resolve()) if a.smoke_run else None,
                    reuse_model_run=str(a.reuse_model_run.resolve()) if a.reuse_model_run else None)
        identity=dict(launch=launch,inputs=inputs,build=build,references=refs,runtime=runtime,variants=variants(),disk_sha256=DISK_SHA256,
                      smoke_gate_sha256=sha256(a.smoke_run/'regression_gate.json') if a.smoke_run and not a.smoke else None,
                      model_cache_key=template_key,
                      no_cad_video_arithmetic='upstream in-place gradient and lower-triangle Hessian; v2 compatibility restoration',
                      protocol='first GT init excluded; all remaining frames at 5cm/5deg; save estimate and count failure before same-frame GT reset',
                      timing='parallel accuracy; decode/metrics/reset/setup excluded; no formal speed claim')
        if previous and previous['identity']!=identity:raise ValueError('Resume identity changed')
        manifest=dict(status='running',pid=os.getpid(),output=str(a.output),identity=identity,expected_jobs=len(jobs)*9,
                      expected_reference_jobs=len(jobs)*2 if a.smoke else 0,completed_jobs=list(previous['completed_jobs']) if previous else [])
        write_json(manifest_path,manifest);done=set(manifest['completed_jobs']);records=[];regressions=[]
        try:
            for number,job in enumerate(jobs):
                config=job['config'];order=variants();offset=number%len(order);order=order[offset:]+order[:offset]
                prepared=a.output/'inputs'/(config['id']+'.json');write_job(job,prepared)
                model_cache=a.cache/'models'/template_key/job['model_fingerprint'][:16]
                for variant in order:
                    key=config['id']+'/'+variant['id'];folder=a.output/'configurations'/key;folder.mkdir(parents=True,exist_ok=True)
                    manifest.update(current=key,job=len(records)+1);write_json(manifest_path,manifest)
                    print('[%d/%d] %s'%(len(records)+1,manifest['expected_jobs'],key),flush=True)
                    if key not in done:
                        command=['xvfb-run','-a','-s','-screen 0 1280x1024x24',str(HERE/'build/evaluate_ablation'),'--input',str(prepared),
                                 '--cache',str(model_cache),'--output',str(folder),'--disk-model',str(a.disk_model),'--flags',variant['flags'],'--variant',variant['id']]
                        native(command,folder)
                    result,times=analyze(folder,job,variant);result['variant']=variant['id'];records.append((result,times))
                    if key not in done:done.add(key);manifest['completed_jobs'].append(key)
                    manifest['completed_evaluated_frames']=sum(r['frames'] for r,t in records);write_json(manifest_path,manifest)
                    print('  success %.4f%%; audit differences %d'%(100*result['successes']/result['frames'],len(result['audit_disagreements'])),flush=True)
                if a.smoke:
                    for target,method in [('A0','m3t_original_rgb'),('A1','ours')]:
                        folder=a.output/'references'/config['id']/method;folder.mkdir(parents=True,exist_ok=True)
                        command,ref_job=reference_command(a,job,method,folder,a.cache/'references'/job['model_fingerprint'][:16]/method)
                        # Reference artifacts have their own completion marker so interrupted
                        # smoke runs never accept a partially written CSV as complete.
                        marker=folder/'complete.json'
                        if not marker.exists():native(command,folder)
                        analyze(folder,ref_job);write_json(marker,dict(status='complete'))
                        comparison=regression(a.output/'configurations'/config['id']/target,folder)
                        regressions.append(dict(config_id=config['id'],target=target,reference=method,**comparison))
                        write_json(a.output/'regression.json',regressions)
                        print('  regression %s vs %s: %s'%(target,method,'PASS' if comparison['passed'] else 'REVIEW REQUIRED'),flush=True)
            summarize(records,a.output,a.lane,identity,a.smoke)
            if a.smoke:
                passed=len(regressions)==len(jobs)*2 and all(r['passed'] for r in regressions)
                write_json(a.output/'regression_gate.json',dict(passed=passed,comparisons=regressions,scope='representative smoke; not proof of full-dataset numerical identity'))
                if not passed:raise ValueError('A0/A1 smoke regression differs; full is blocked until the cause is understood. See regression.json')
            manifest.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write_json(manifest_path,manifest)
        except BaseException as e:
            manifest.update(status='failed',error=str(e));write_json(manifest_path,manifest);raise
    print('Complete: '+str(a.output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lane',choices=LANES,required=True)
    p.add_argument('--dataset',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kit',type=Path,default=Path('/home/jyj/Projects/industrial-tracker-comparison'))
    p.add_argument('--rbot-kit',type=Path,default=Path('/home/jyj/Projects/rbot-tracker-comparison'))
    p.add_argument('--disk-model',type=Path,default=Path('/home/jyj/Projects/M3T/disk_512.onnx'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--smoke-run',type=Path)
    p.add_argument('--reuse-model-run',type=Path)
    a=p.parse_args();a.dataset=a.dataset or Path(LANES[a.lane]['root']);a.cache=a.cache or Path('/home/jyj/autodl-tmp/M3T-ablation-cache')/a.lane
    run(a)
