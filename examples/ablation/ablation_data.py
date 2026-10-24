"""Read-only dataset adapters and one immutable external-sequence job per target."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'support'))
if not (HERE/'support/rbot_data.py').exists():
    sys.path.insert(0,str(HERE.parent/'industrial'))
    sys.path.insert(0,str(HERE.parent/'rbot_compare'))
import dataset as base
import rbot_data as rbot
from rbot_data import sha256,write_json,DISK_SHA256

LANES={
    'rbot':dict(gpu=1,root='/home/jyj/autodl-tmp/RBOT-dataset',configurations=72,frames=1001),
    'industrial_base':dict(gpu=4,root='/home/jyj/autodl-tmp/industrial-render-900x600/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016',configurations=150,frames=500),
    'industrial_advanced':dict(gpu=7,root='/home/jyj/autodl-tmp/industrial-render-advanced-v1/dataset/交付物/工业跟踪进阶数据集-v1.0-20261018',configurations=324,frames=500)}
K_BASE=dict(fx=1125,fy=1125,cx=449.5,cy=299.5,width=900,height=600)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def variants():
    value=load(HERE/'variants.json')
    if len(value)!=9 or [v['id'] for v in value]!=['A%d'%i for i in range(9)]:raise ValueError('Expected the frozen nine-variant design')
    return value


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def safe(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve();p.relative_to(root)
    if p==root:raise ValueError('Empty dataset path')
    return p


def advanced_index(root):
    state=root.parent.parent/'中间结果文件/run_state_advanced_v1'
    final=load(state/'validation/final_audit.json');inputs=load(state/'independent_audit/inputs.json');scenes=load(state/'independent_audit/scenes.json')
    if (final.get('status'),final.get('scope'),final.get('images'))!=('passed','complete_dataset',54000):raise ValueError('Complete advanced rendering audit required')
    if inputs.get('status')!='passed' or (scenes.get('status'),scenes.get('scope'),scenes.get('sequences_checked'),scenes.get('total_frame_checks'))!=('passed','all integer frames',108,54000):
        raise ValueError('Complete independent input and scene audits required')
    path=state/'independent_audit/evaluation_index.json';rows=load(path)['configurations']
    if len(rows)!=324 or len({r['sequence_id']+'__'+r['object'] for r in rows})!=324 or sum(bool(r['is_focus']) for r in rows)!=108:
        raise ValueError('Unexpected advanced scope')
    camera=load(root/'cameras/camera.json')
    for key,value in dict(width=900,height=600,frames_per_sequence=500,K=[[1125,0,449.5],[0,1125,299.5],[0,0,1]],translation_unit='meter',pose_direction='p_camera = R @ p_object + t').items():
        if camera.get(key)!=value:raise ValueError('Advanced calibration mismatch: '+key)
    return rows,{str(p.relative_to(state)):sha256(p) for p in [path,state/'validation/final_audit.json',state/'independent_audit/inputs.json',state/'independent_audit/scenes.json']}


def select(lane,root,smoke):
    if lane=='rbot':
        return rbot.configurations(['ape','driller'] if smoke else None,['a_regular','c_noisy'] if smoke else None),{}
    if lane=='industrial_base':
        base.read_camera(root)
        return (base.configurations(['scene_01'],['camera_motion','camera_and_objects_motion'],['01_normal','05_sensor_noise'],['obj_000001']) if smoke else base.configurations()),{}
    rows,audit=advanced_index(root)
    if smoke:rows=[r for r in rows if r['scene']=='scene_04' and r['trajectory']=='t01' and r['condition'] in ['B00','O03','M02'] and r['is_focus']]
    return [dict(r,id=r['sequence_id']+'__'+r['object'],motion=r['motion_mode']) for r in rows],audit


def protect(stack,lane,root,hold):
    if lane=='rbot':return
    runtime=root.parent.parent/'中间结果文件'
    states=['run_state','run_state_v09'] if lane=='industrial_base' else ['run_state_advanced_v1']
    for state in states:
        path=runtime/state/'render.lock'
        if path.exists():hold(stack,path,shared=True)


def prepare(lane,root,cache,smoke):
    root=Path(root).resolve();configs,audit=select(lane,root,smoke);frames=21 if smoke else LANES[lane]['frames']
    if not configs:raise ValueError('Empty selection')
    if not smoke and len(configs)!=LANES[lane]['configurations']:raise ValueError('Incorrect full configuration count')
    files={};jobs=[];images=set();model_keys={}
    if lane=='rbot':
        validation,poses=rbot.preflight(root,configs,frames)
        files.update(validation['asset_sha256']);files['poses_first.txt']=validation['pose_sha256']
        normalized=Path(cache)/'inputs'/validation['pose_sha256'][:16]/'poses_first_m.txt';rbot.write_normalized_poses(normalized,poses)
        fingerprint=validation['input_fingerprint']
    else:
        files['cameras/camera.json']=sha256(root/'cameras/camera.json')
        if lane=='industrial_advanced':files['cameras/sequence_index.csv']=sha256(root/'cameras/sequence_index.csv')
    for c in configs:
        if lane=='rbot':
            directory=root/c['body']/'frames';mesh=root/c['body']/(c['body']+'.obj');pose_file=normalized;k=rbot.INTRINSICS
            prefix=c['sequence'];first=0;digits=4;scale=.001;rows=1001
        else:
            directory,pose_file,mesh=base.paths(root,c) if lane=='industrial_base' else (
                safe(root,c['rgb_path']),safe(root,c['ground_truth_file']),root/'objects'/c['object']/(c['object']+'.obj'))
            base.read_poses(pose_file);files[pose_file.relative_to(root).as_posix()]=sha256(pose_file)
            # Same OBJ/MTL/diffuse-map parser as RBOT, rooted at objects/.
            if c['object'] not in model_keys:
                assets={'objects/'+name:digest for name,digest in rbot.model_assets(root/'objects',c['object']).items()}
                files.update(assets);model_keys[c['object']]=signature(assets)
            k=K_BASE;prefix='';first=1;digits=6;scale=1.;rows=500
            for i in range(1,frames+1):
                p=directory/('%06d.png'%i)
                if p in images:continue
                header=base.png_header(p);stat=p.stat();files[p.relative_to(root).as_posix()]=[stat.st_size,stat.st_mtime_ns,hashlib.sha256(header).hexdigest()];images.add(p)
        job=dict(config=c,dataset=str(root),object=c['object'],sequence=c['sequence'] if lane=='rbot' else c['scene']+'__'+c['motion']+'__'+c['condition'],
                 images=str(directory),mesh=str(mesh),poses=str(pose_file),intrinsics=k,frames=frames,pose_rows=rows,first_image=first,
                 leading_zeros=digits,prefix=prefix,geometry_scale=scale,input_bits=8 if lane=='rbot' else 16)
        job['model_fingerprint']=validation['model_fingerprints'][c['object']] if lane=='rbot' else model_keys[c['object']]
        jobs.append(job)
    if lane!='rbot':fingerprint=signature(dict(files=files,audits=audit,root=str(root)))
    identity=dict(dataset=str(root),lane=lane,fingerprint=fingerprint,source_assets={n:v for n,v in files.items() if isinstance(v,str)},
                  audit_sha256=audit,frames=frames,configurations=len(jobs),input_scope='poses/CAD/metadata full SHA256; selected PNG headers, sizes, mtimes; full decode online')
    return jobs,identity


def write_job(job,path):
    write_json(path,{k:v for k,v in job.items() if k not in ['config','model_fingerprint']})
