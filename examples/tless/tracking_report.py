"""Independent per-frame symmetry audit and complete paired T-LESS reports."""
import csv
import math
from pathlib import Path
import numpy as np
from tracking_data import METHODS,LABELS,load,metric,rigid,save

TIMES=['decode_us','preprocessing_us','tracking_wall_us','compute_wall_us','metric_us','reset_us',
       'frame_wall_us','cycle_us','correspondences_us','gradient_hessian_us','optimization_us','results_us']


def matrix(row,prefix):
    p=np.eye(4);p[:3,:3]=[[float(row['%s_r%d%d'%(prefix,r,c)]) for c in range(3)] for r in range(3)]
    p[:3,3]=[float(row[prefix+'_t'+a]) for a in 'xyz'];return p


def analyze(folder,config,index,prepared,root,frames,model_cache,scene_cache):
    meta=load(folder/'metadata.json')
    if (meta['method']!=folder.name or meta['config_id']!=config['id'] or meta['frame_count']!=frames-1 or meta['initial_image_id']!=config['image_ids'][0]
            or meta['estimate_capture']!='before_gt_reset' or abs(meta['mssd_reset_fraction']-.1)>1e-12
            or meta['cad_texture_available']!=0 or meta['cad_disk_executed']!=0):raise ValueError('Unexpected native metadata')
    obj=config['obj_id'];name='obj_%06d'%obj;model=index['models'][str(obj)]
    if meta.get('cad_disk_configured')!=int(folder.name=='ours'):raise ValueError('CAD configuration metadata mismatch')
    k=np.array(index['intrinsics'])
    if not np.allclose(meta.get('intrinsics',[]),[k[0,0],k[1,1],k[0,2],k[1,2],*index['image_size']],atol=1e-4,rtol=0):raise ValueError('Native intrinsics metadata mismatch')
    if abs(meta['diameter_m']-model['diameter_m'])>1e-10 or meta['symmetry_count']!=model['symmetries']:raise ValueError('Native scoring model mismatch')
    if obj not in model_cache:
        verts=np.loadtxt(str(prepared/'models'/name/'evaluation.xyz'));syms=np.loadtxt(str(prepared/'models'/name/'symmetries.txt'),ndmin=2)
        transforms=np.repeat(np.eye(4)[None],len(syms),axis=0);transforms[:,:3,:3]=syms[:,:9].reshape(-1,3,3);transforms[:,:3,3]=syms[:,9:]
        model_cache[obj]=(verts,transforms)
    if config['scene_id'] not in scene_cache:
        source=root/'test_primesense'/('%06d'%config['scene_id'])
        scene_cache[config['scene_id']]=(load(source/'scene_gt.json'),load(source/'scene_gt_info.json'))
    gt,vis=scene_cache[config['scene_id']];verts,syms=model_cache[obj]
    with (folder/'poses.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=frames-1:raise ValueError('Missing/extra CSV rows')
    segment=0;differences=[];errors=[];recomputed=[];timing=[]
    for i,row in enumerate(rows):
        image=config['image_ids'][i+1];expected=gt[str(image)][config['gt_index']]
        reference=rigid(expected['cam_R_m2c'],expected['cam_t_m2c']);actual=matrix(row,'gt')
        if int(row['frame'])!=i or int(row['image_id'])!=image or int(row['segment_id'])!=segment:raise ValueError('Image/reset alignment mismatch')
        if not np.allclose(reference,actual,atol=1e-7,rtol=0):raise ValueError('CSV truth does not match input instance')
        v=vis[str(image)][config['gt_index']]['visib_fract']
        if abs(float(row['visibility'])-v)>1e-10:raise ValueError('Visibility mismatch')
        est=matrix(row,'est');e3,e2=metric(est,actual,verts,syms,k)
        native3,native2=float(row['mssd_m']),float(row['mspd_px'])
        for a,b,tol in [(native3,e3,2e-7),(native2,e2,0.003)]:
            if math.isinf(a) and math.isinf(b):continue
            if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tol:raise ValueError('Native and independent symmetry errors disagree')
        success=int(row['success']);again=int(math.isfinite(e3) and e3<=.1*model['diameter_m'])
        if (success not in [0,1] or success!=int(math.isfinite(native3) and native3<=.1*model['diameter_m'])
                or int(row['pose_finite'])!=int(math.isfinite(e3) and math.isfinite(e2))
                or int(row['reset_after'])!=1-success):raise ValueError('Reset classification mismatch')
        if success!=again:differences.append(image)
        detail=vis[str(image)][config['gt_index']];x,y,w,h=detail.get('bbox_visib',[8,8,1,1])
        boundary=bool(w>0 and h>0 and (x<8 or y<8 or x+w>712 or y+h>532))
        segment+=1-success;errors.append((native3/model['diameter_m'],native2,v,int(row['pose_finite']),boundary))
        recomputed.append(again)
        t={name:float(row[name]) for name in TIMES}
        if any(not math.isfinite(x) or x<0 for x in t.values()):raise ValueError('Invalid timing')
        tolerance=max(2.,t['tracking_wall_us']*1e-5)
        if (abs(t['tracking_wall_us']-t['decode_us']-t['compute_wall_us'])>tolerance
                or abs(t['frame_wall_us']-t['tracking_wall_us']-t['metric_us']-t['reset_us'])>tolerance
                or t['preprocessing_us']>t['compute_wall_us']+2):raise ValueError('Timing decomposition mismatch')
        timing.append(t)
    result=dict(config_id=config['id'],scene_id=config['scene_id'],obj_id=obj,elevation=config['elevation'],gt_index=config['gt_index'],
                frames=len(rows),successes=sum(int(r['success']) for r in rows),resets=segment,
                recomputed_successes=sum(recomputed),audit_disagreements=differences,invalid=sum(not e[3] for e in errors),
                cad_disk_executed=False,pose_csv=str(folder/'poses.csv'))
    save(folder/'audit.json',result)
    return result,errors,timing


def stats(values):
    a=np.asarray(values,dtype=float)
    def safe(v):return float(v) if math.isfinite(v) else None
    return dict(frames=len(a),mean=safe(a.mean()),median=safe(np.median(a)),p95=safe(np.percentile(a,95)))


def summarize(results,index,output,smoke):
    keys=[(r[0]['config_id'],r[0]['method']) for r in results]
    configs={key[0] for key in keys}
    if not configs or len(set(keys))!=len(keys) or set(keys)!={(c,m) for c in configs for m in METHODS}:
        raise ValueError('Unpaired/incomplete results')
    groups={m:[r for r in results if r[0]['method']==m] for m in METHODS};methods={}
    for method,records in groups.items():
        counts=[r[0] for r in records];errs=[v for r in records for v in r[1]]
        times=[v for r in records for v in r[2][10:]]
        if not records:raise ValueError('Incomplete methods')
        frames=sum(c['frames'] for c in counts);successes=sum(c['successes'] for c in counts)
        def subset(lo,hi):
            selected=[e for e in errs if lo<=e[2]<hi]
            return dict(frames=len(selected),success_percent=100*sum(e[0]<=.1 for e in selected)/len(selected) if selected else None)
        def edge_subset(flag):
            selected=[e for e in errs if e[4]==flag]
            return dict(frames=len(selected),success_percent=100*sum(e[0]<=.1 for e in selected)/len(selected) if selected else None)
        methods[method]=dict(configurations=len(records),frames=frames,successes=successes,resets=sum(c['resets'] for c in counts),
                             success_percent=100*successes/frames,recomputed_success_percent=100*sum(c['recomputed_successes'] for c in counts)/frames,
                             invalid=sum(c['invalid'] for c in counts),audit_disagreements=sum(len(c['audit_disagreements']) for c in counts),
                             mssd_fraction=stats([e[0] for e in errs]),mspd_px=stats([e[1] for e in errs]),
                             recall_mssd_fraction={str(t):100*sum(e[0]<=t for e in errs)/frames for t in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]},
                             recall_mspd_px={str(t):100*sum(e[1]<=t for e in errs)/frames for t in range(5,51,5)},
                             visibility={'under_10_percent':subset(0,.1),'10_to_50_percent':subset(.1,.5),'50_to_100_percent':subset(.5,1.000001)},
                             native_boundary={'within_8px':edge_subset(True),'interior':edge_subset(False)},
                             timing={t:stats([r[t] for r in times]) for t in TIMES},cad_disk_executed=False)
    value=dict(status='complete',dataset='T-LESS static viewpoint rings',smoke=smoke,protocol=index['protocol'],
               index_fingerprint=index['fingerprint'],configurations=[r[0] for r in results],methods=methods,
               primary_timing_metric='compute_wall_us',timing_note='Includes actual online lossless padding; excludes PNG decode, metrics, GT reset and setup',
               scoring_note='MSSD on all provided models_eval vertices and finite BOP-style symmetry set; native-pixel MSPD; not official BOP AR',
               capability_note='Textureless CAD; CAD-DISK configured in ours but executes no matching branch')
    save(output/'summary.json',value)
    lines=['# T-LESS五方法静态视点序列评测','',
           '%d个目标视点环、%d个方法作业。GT首帧初始化；MSSD/直径≤0.1成功，失败帧计入后按本帧GT重置。'%(len(configs),len(results)),'',
           '输入原生720×540，在线整数平移与127灰色填充为757×635，无缩放/插值/裁切；计算时间包含这项预处理，排除PNG解码。','',
           '**CAD无UV/贴图，我们的CAD-DISK匹配分支未参与。结果不是完整外观分支有效性的证明，也不是BOP定位排行榜成绩。**','',
           '| 方法 | 评价帧 | MSSD成功率 | 重算成功率 | 无效估计 | 计算均值ms | 计算P95ms | 计算FPS |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=methods[m];t=v['timing']['compute_wall_us'];lines.append('| %s | %d | %.4f%% | %.4f%% | %d | %.3f | %.3f | %.2f |'%(LABELS[m],v['frames'],v['success_percent'],v['recomputed_success_percent'],v['invalid'],t['mean']/1000,t['p95']/1000,1e6/t['mean']))
    lines+=['','计算FPS为计时均值倒数，不代表完整在线吞吐；每配置前10帧仅从时间剔除。所有帧均参与主评价，低可见度仅分组报告，未从跟踪过程中删除。',
            '','## 可见度分组','','| 可见度 | '+' | '.join(LABELS[m] for m in METHODS)+' |','|---|'+'---:|'*5]
    for group in ['under_10_percent','10_to_50_percent','50_to_100_percent']:
        cells=[]
        for m in METHODS:
            v=methods[m]['visibility'][group];cells.append('%.4f%% (%d)'%(v['success_percent'],v['frames']) if v['frames'] else '—')
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines+=['','## 原生视场边界分组','','边界指GT可见框距原始720×540图像边缘不足8px；填充区为未知观测的统一灰色近似，基线没有公共无效像素掩码接口，边界统计应单独检查。','',
            '| 原生边界 | '+' | '.join(LABELS[m] for m in METHODS)+' |','|---|'+'---:|'*5]
    for group in ['within_8px','interior']:
        cells=[]
        for m in METHODS:
            v=methods[m]['native_boundary'][group];cells.append('%.4f%% (%d)'%(v['success_percent'],v['frames']) if v['frames'] else '—')
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines+=['','## 全部配对配置','','| 配置 | '+' | '.join(LABELS[m] for m in METHODS)+' |','|---|'+'---:|'*5]
    mapping={(r[0]['config_id'],r[0]['method']):r[0] for r in results}
    for c in sorted(configs):lines.append('| '+c+' | '+' | '.join('%.4f%%'%(100*mapping[c,m]['successes']/mapping[c,m]['frames']) for m in METHODS)+' |')
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value
