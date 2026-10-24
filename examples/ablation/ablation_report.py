"""Paired component accuracy; all concurrent timings are explicitly descriptive."""
import numpy as np
from ablation_audit import TIMES
from ablation_data import variants,write_json


def stats(values):
    return dict(frames=len(values),mean_us=float(np.mean(values)),median_us=float(np.median(values)),p95_us=float(np.percentile(values,95)))


def summarize(records,output,lane,identity,smoke):
    design=variants();ids=[v['id'] for v in design];keys=[(r['id'],r['variant']) for r,t in records];configs={k[0] for k in keys}
    if not configs or len(set(keys))!=len(keys) or set(keys)!={(c,v) for c in configs for v in ids}:raise ValueError('Incomplete paired ablation')
    methods={}
    def accuracy(rows):
        frames=sum(r['frames'] for r in rows);success=sum(r['successes'] for r in rows)
        return dict(configurations=len(rows),frames=frames,successes=success,success_percent=100*success/frames,
                    recomputed_success_percent=100*sum(r['recomputed_successes'] for r in rows)/frames,
                    resets=sum(r['resets'] for r in rows),invalid=sum(r['invalid'] for r in rows),
                    audit_disagreements=sum(len(r['audit_disagreements']) for r in rows))
    for v in design:
        chosen=[(r,t) for r,t in records if r['variant']==v['id']];rows=[r for r,t in chosen]
        timed=np.concatenate([t[10:] for r,t in chosen])
        if not len(timed):raise ValueError('No post-warmup timing samples')
        result=accuracy(rows);result.update(v,timing={name:stats(timed[:,j]) for j,name in enumerate(TIMES)})
        groups={}
        for key in ['condition','motion','object','scene','trajectory']:
            available=sorted({r[key] for r in rows if key in r})
            if available:groups[key]={name:accuracy([r for r in rows if r.get(key)==name]) for name in available}
        if any('is_focus' in r for r in rows):groups['focus']={'focus':accuracy([r for r in rows if r.get('is_focus')])}
        result['groups']=groups;methods[v['id']]=result
    full=methods['A1']['success_percent']
    for result in methods.values():result['delta_vs_full_pp']=result['success_percent']-full
    summary=dict(status='complete',lane=lane,smoke=smoke,identity=identity,variants=methods,configurations=[r for r,t in records],
                 primary_metric='5cm/5deg success; first image GT-init excluded, failure counted before same-frame GT reset',
                 timing_scope='shared-server concurrent accuracy run; compute excludes PNG decode and offline scoring; not formal speed comparison',
                 smoothing='same Full two-velocity smoothing for P/T; T switch controls only prior application',
                 fusion='one combined reliability and information-complement module; no independent mechanism claims')
    write_json(output/'summary.json',summary)
    lines=['# '+lane+'：九组消融','',
           '同一冻结代码基底；Region与视频ORB始终保留。GT初始化/失败后GT重置；5cm/5°；预热10帧仅从计时排除。',
           '','**本轮与其他任务并行，耗时仅作共享负载记录，不作为正式速度结论。**','',
           '| 组 | 变体 | D/F/P/T/N | 成功率 | 相对Full百分点 | 重置 | 无效估计 | 计算均值ms（负载下） |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for v in design:
        r=methods[v['id']];lines.append('| %s | %s | %s | %.4f%% | %+.4f | %d | %d | %.3f |'%(v['id'],v['name'],v['flags'],r['success_percent'],r['delta_vs_full_pp'],r['resets'],r['invalid'],r['timing']['compute_wall_us']['mean_us']/1000))
    lines+=['','F同时包含可靠性权重和信息方向互补，本轮不能将两者的单独贡献拆开解释。所有组保留失败帧及无效估计的分母。',
            '','## 按条件成功率','','| 条件 | '+' | '.join(ids)+' |','|---|'+'---:|'*len(ids)]
    conditions=sorted({r['condition'] for r,t in records})
    for c in conditions:lines.append('| '+c+' | '+' | '.join('%.4f%%'%methods[v]['groups']['condition'][c]['success_percent'] for v in ids)+' |')
    lines+=['','## 全部配对配置','','| 配置 | '+' | '.join(ids)+' |','|---|'+'---:|'*len(ids)]
    mapping={(r['id'],r['variant']):r for r,t in records}
    for c in sorted(configs):lines.append('| '+c+' | '+' | '.join('%.4f%%'%(100*mapping[c,v]['successes']/mapping[c,v]['frames']) for v in ids)+' |')
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
