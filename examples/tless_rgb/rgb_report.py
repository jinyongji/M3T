"""Reuse the frozen pose audit; additionally audit experiment switches/support."""
import csv
import math
from pathlib import Path
import numpy as np

def analyze(folder,config,index,prepared,dataset,frames,variant,models,scenes,data,base):
    meta=data.load(folder/'metadata.json')
    expected=dict(rgb_variant=variant['id'],rgb_video_verification=int(variant['video_verify']),
                  rgb_visibility_consensus=int(variant['visibility']),rgb_region_control=int(variant['region_control']))
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('RGB runtime flags differ from the planned variant')
    record,errors,times=base.analyze(folder,config,index,prepared,dataset,frames,models,scenes)
    totals=dict(candidates=0,retained=0,verified_frames=0,rejected_frames=0)
    with (folder/'poses.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            candidates=int(row['rgb_video_candidates']);retained=int(row['rgb_video_retained']);verified=int(row['rgb_video_verified'])
            evidence=float(row['rgb_visibility_evidence']);activation=float(row['rgb_visibility_activation'])
            if not 0<=retained<=candidates or verified not in [0,1] or not all(math.isfinite(x) and 0<=x<=1.00001 for x in [evidence,activation]):raise ValueError('Invalid causal RGB diagnostics')
            if variant['region_control']:
                if retained or verified:raise ValueError('Region control retained video constraints')
            elif variant['video_verify']:
                if verified and (retained<6 or 2*retained<candidates):raise ValueError('Insufficient verified support')
                if not verified and retained:raise ValueError('Rejected video constraints retained')
            elif retained!=candidates or verified:raise ValueError('Disabled verification changed support')
            if not variant['visibility'] and activation:raise ValueError('Visibility consensus unexpectedly active')
            totals['candidates']+=candidates;totals['retained']+=retained
            totals['verified_frames']+=verified;totals['rejected_frames']+=int(variant['video_verify'] and not verified)
    record.update(variant=variant['id'],rgb_geometry=totals)
    return record,np.asarray(errors,dtype=float),np.asarray([[r[k] for k in base.TIMES] for r in times],dtype=float)

def regression(left,right,base):
    def read(folder):
        with (folder/'poses.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    a,b=read(left),read(right)
    if len(a)!=len(b):raise ValueError('RGB regression frame mismatch')
    differences=[];maximum_r=maximum_t=0.;first=None
    for i,(x,y) in enumerate(zip(a,b)):
        if x['image_id']!=y['image_id']:raise ValueError('RGB regression image mismatch')
        p,q=base.matrix(x,'est'),base.matrix(y,'est')
        r=float(np.abs(p[:3,:3]-q[:3,:3]).max());t=float(np.linalg.norm(p[:3,3]-q[:3,3]))
        flags=any(x[k]!=y[k] for k in ['success','reset_after','segment_id','pose_finite'])
        finite=bool(np.isfinite(p).all() and np.isfinite(q).all())
        if flags or not finite:differences.append(i)
        if finite:maximum_r=max(maximum_r,r);maximum_t=max(maximum_t,t)
        if first is None and (flags or not finite or r>1e-4 or t>5e-5):first=dict(row_index=i,image_id=x['image_id'],rotation_entry=r if finite else None,translation_m=t if finite else None,classification_or_segment_difference=flags)
    return dict(passed=not differences and maximum_r<=1e-4 and maximum_t<=5e-5,frames=len(a),maximum_rotation_entry=maximum_r,maximum_translation_m=maximum_t,
                classification_or_invalid_rows=differences,first_difference=first,tolerance_rotation_entry=1e-4,tolerance_translation_m=5e-5)

def summarize(records,variants,output,stage,identity,data,base):
    keys={(r['config_id'],r['variant']) for r,e,t in records};configs={c for c,v in keys}
    if len(keys)!=len(records) or keys!={(c,v['id']) for c in configs for v in variants}:raise ValueError('Incomplete paired RGB results')
    methods={}
    for variant in variants:
        selected=[(r,e,t) for r,e,t in records if r['variant']==variant['id']]
        frames=sum(r['frames'] for r,e,t in selected);successes=sum(r['successes'] for r,e,t in selected)
        errors=np.concatenate([e for r,e,t in selected]);timing=np.concatenate([t[10:] for r,e,t in selected])
        visibility={}
        for label,lo,hi in [('under_10_percent',0,.1),('10_to_50_percent',.1,.5),('50_to_100_percent',.5,1.000001)]:
            part=errors[(errors[:,2]>=lo)&(errors[:,2]<hi)]
            visibility[label]=dict(frames=len(part),success_percent=float(100*np.mean(part[:,0]<=.1)) if len(part) else None)
        methods[variant['id']]=dict(name=variant['name'],configurations=len(selected),frames=frames,successes=successes,resets=sum(r['resets'] for r,e,t in selected),
            success_percent=100*successes/frames,invalid=sum(r['invalid'] for r,e,t in selected),audit_disagreements=sum(len(r['audit_disagreements']) for r,e,t in selected),
            timing={k:base.stats(timing[:,i]) for i,k in enumerate(base.TIMES)},visibility=visibility,
            rgb_geometry={k:sum(r['rgb_geometry'][k] for r,e,t in selected) for k in ['candidates','retained','verified_frames','rejected_frames']})
    data.save(output/'summary.json',dict(status='complete',stage=stage,identity=identity,variants=methods,configurations=[r for r,e,t in records],
        note='Exploratory RGB change after inspection of full T-LESS results; not an independent unseen-test claim; CAD-DISK remains inactive'))
    lines=['# T-LESS RGB改进配对实验','', '阶段：'+stage+'。MSSD/d≤0.1；原视点、初始化、GT重置、所有可见度与计算计时协议保留。CAD-DISK仍不参与。','',
        '| 变体 | 配置 | 成功率 | 相对B0/百分点 | 计算均值ms | P95ms |','|---|---|---:|---:|---:|---:|']
    for k,v in methods.items():
        t=v['timing']['compute_wall_us'];lines.append('| %s | %s | %.4f%% | %+.4f | %.3f | %.3f |'%(k,v['name'],v['success_percent'],v['success_percent']-methods['B0']['success_percent'],t['mean']/1000,t['p95']/1000))
    lines+=['','B4只移除视频约束，仍做纹理维护，不能把其耗时称为纯Region算法速度。共享服务器单轮耗时不作为正式速度结论。','',
        '## 可见度分组','','| 可见度 | '+' | '.join(v['id'] for v in variants)+' |','|---|'+'---:|'*len(variants)]
    for label in methods['B0']['visibility']:
        lines.append('| '+label+' | '+' | '.join('%.4f%% (%d)'%(methods[v['id']]['visibility'][label]['success_percent'],methods[v['id']]['visibility'][label]['frames']) if methods[v['id']]['visibility'][label]['frames'] else '—' for v in variants)+' |')
    lines+=['','## 全部配对配置','','| 配置 | '+' | '.join(v['id'] for v in variants)+' |','|---|'+'---:|'*len(variants)]
    pairs={c:{r['variant']:r for r,e,t in records if r['config_id']==c} for c in configs}
    for c,g in sorted(pairs.items()):lines.append('| '+c+' | '+' | '.join('%.4f%%'%(100*g[v['id']]['successes']/g[v['id']]['frames']) for v in variants)+' |')
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
