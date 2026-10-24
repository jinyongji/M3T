"""Audit an exported full RBOT summary and write a paper-oriented Markdown report.

This tool reads completed results only. It never starts trackers or rewrites inputs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

METHODS = ['rbgt', 'srt3d', 'icg_rgb', 'm3t_original_rgb', 'ours']
LABELS = dict(zip(METHODS, ['RBGT', 'SRT3D', 'ICG', 'ICG+', 'M3T（我们这版）']))
SEQUENCES = ['a_regular', 'b_dynamiclight', 'c_noisy', 'd_occlusion']
SEQUENCE_NAMES = ['常规', '动态光照', '噪声', '遮挡']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def close(a, b, label, tolerance=1e-7):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tolerance:
        raise ValueError('%s differs: %r versus %r' % (label, a, b))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def validate(value, source_report):
    if value['status'] != 'complete' or value['dataset'] != 'RBOT' or set(value['methods']) != set(METHODS):
        raise ValueError('Need a completed five-method RBOT result')
    if value['primary_timing_metric'] != 'compute_wall_us' or value['identity']['frames'] != 1001:
        raise ValueError('Need full-length compute-wall timing')
    if value['identity']['warmup'] != 10 or value['identity']['repeats'] != 1:
        raise ValueError('This report describes one repeat with ten timing warmup frames')
    if value['thresholds'] != dict(translation_m=0.05, rotation_deg=5):
        raise ValueError('Unexpected thresholds')
    paired = {p['id']: p for p in value['paired_configurations']}
    rows = value['configurations']
    if len(paired) != 72 or len(rows) != 360:
        raise ValueError('Need 72 complete configurations and 360 jobs')
    body_names = sorted({p['body'] for p in paired.values()})
    if len(body_names) != 18 or {p['sequence'] for p in paired.values()} != set(SEQUENCES):
        raise ValueError('Unexpected object/sequence scope')
    seen = set(); discrepancies = []
    for row in rows:
        key = row['id'], row['method']
        if key in seen or row['frames'] != 1000 or row['repeat'] != 1:
            raise ValueError('Duplicate or incomplete job: '+str(key))
        seen.add(key)
        if row['successes']+row['reset_count'] != row['frames']:
            raise ValueError('Success/failure denominator mismatch')
        close(row['success_percent'], 100*row['successes']/row['frames'], str(key))
        close(row['success_percent'], paired[row['id']]['success_percent'][row['method']], str(key))
        if row['success_classification_disagreement_count'] != len(row['success_classification_disagreement_images']):
            raise ValueError('Disagreement image count mismatch')
        if row['success_classification_disagreement_count']:
            discrepancies.append(row)
    for identifier in paired:
        if {(identifier, m) for m in METHODS} - seen:
            raise ValueError('Missing paired method')
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        total = value['methods'][method]
        for key in ['frames', 'successes', 'reset_count', 'invalid_estimate_count', 'success_classification_disagreement_count']:
            if sum(r[key] for r in selected) != total[key]:
                raise ValueError('Aggregate mismatch: '+method+'/'+key)
        if total['frames'] != 72000 or total['configurations'] != 72:
            raise ValueError('Unexpected full-run denominator')
        close(total['success_percent'], 100*total['successes']/total['frames'], method)
        close(total['recomputed_success_percent'], 100*sum(r['recomputed_successes'] for r in selected)/total['frames'], method)
        timing = total['timing']
        for key in timing:
            if timing[key]['frames'] != 71280:
                raise ValueError('Unexpected timing denominator: '+method+'/'+key)
        close(timing['compute_wall_us']['mean_us'], timing['tracking_wall_us']['mean_us']-timing['decode_us']['mean_us'], method)
        close(total['derived_compute_fps'], 1e6/timing['compute_wall_us']['mean_us'], method)
        for group_name, field in [('by_sequence', 'sequence'), ('by_body', 'body')]:
            for group, aggregate in total[group_name].items():
                members = [r for r in selected if r[field] == group]
                for key in ['frames', 'successes', 'reset_count']:
                    if sum(r[key] for r in members) != aggregate[key]:
                        raise ValueError('Group mismatch: '+method+'/'+group)
                close(aggregate['success_percent'], 100*aggregate['successes']/aggregate['frames'], method+'/'+group)
    # Validate the numeric cells of the independently supplied Markdown table.
    md_configurations = {}
    for line in source_report.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 7 and cells[1] in SEQUENCES:
            md_configurations[cells[1]+'__'+cells[0]] = [float(v.rstrip('%')) for v in cells[2:]]
    if set(md_configurations) != set(paired):
        raise ValueError('Source Markdown configuration scope differs')
    for identifier, percents in md_configurations.items():
        for method, percent in zip(METHODS, percents):
            close(percent, paired[identifier]['success_percent'][method], identifier+'/'+method, tolerance=5.1e-5)
    return body_names, discrepancies


def object_bootstrap(value, draws=20000, seed=20261023):
    by_body = defaultdict(list)
    for p in value['paired_configurations']:
        by_body[p['body']].append(p['success_percent']['ours']-p['success_percent']['m3t_original_rgb'])
    clusters = [statistics.mean(v) for _, v in sorted(by_body.items())]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(clusters, k=len(clusters))) for _ in range(draws))
    def percentile(q):
        index = q*(len(means)-1); lower = int(index)
        return means[lower]+(means[min(lower+1, len(means)-1)]-means[lower])*(index-lower)
    return dict(unit='percentage points', cluster='object; retain its four sequences together',
                clusters=len(clusters), draws=draws, seed=seed, mean=statistics.mean(clusters),
                percentile_95_interval=[percentile(0.025), percentile(0.975)])


def plot(value, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
    labels = ['RBGT', 'SRT3D', 'ICG', 'ICG+', 'Ours']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    x = np.arange(4); width = 0.15
    for j, method in enumerate(METHODS):
        totals = value['methods'][method]
        axes[0].bar(x+(j-2)*width, [totals['by_sequence'][s]['success_percent'] for s in SEQUENCES],
                    width=width, color=colors[j], label=labels[j])
        axes[1].scatter(totals['timing']['compute_wall_us']['mean_us']/1000, totals['success_percent'],
                        s=85, color=colors[j], edgecolors='white', linewidth=0.7, zorder=3)
        if j >= 3:
            axes[1].annotate(labels[j], (totals['timing']['compute_wall_us']['mean_us']/1000, totals['success_percent']),
                             xytext=(-5, -15), textcoords='offset points', ha='right', fontsize=10)
    axes[0].set_xticks(x, ['Regular', 'Dynamic light', 'Noisy', 'Occlusion'])
    axes[0].set_ylim(0, 105); axes[0].set_ylabel('Tracking success (%)')
    axes[0].set_title('(a) Accuracy by RBOT sequence')
    axes[0].legend(ncol=5, fontsize=8.3, loc='upper center', bbox_to_anchor=(0.5, 1.16), frameon=False)
    axes[1].set_xscale('log'); axes[1].set_xlim(1.1, 45); axes[1].set_ylim(80, 100)
    axes[1].set_xticks([1.5, 3, 6, 12, 24, 40], ['1.5', '3', '6', '12', '24', '40'])
    axes[1].set_xlabel('Mean compute time (ms/frame; log scale)')
    axes[1].set_ylabel('Overall tracking success (%)')
    axes[1].set_title('(b) Accuracy and compute cost')
    for ax in axes:
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=0.22)
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target)+'.png', dpi=200)
    fig.savefig(str(target)+'.svg')
    plt.close(fig)


def create_report(value, bodies, discrepancies, bootstrap, archive, industrial, advanced, figure):
    m = value['methods']; base = m['m3t_original_rgb']; ours = m['ours']
    paired = value['paired_configurations']
    difference = ours['success_percent']-base['success_percent']
    delta = [p['success_percent']['ours']-p['success_percent']['m3t_original_rgb'] for p in paired]
    wtl = [sum(d>1e-9 for d in delta), sum(abs(d)<=1e-9 for d in delta), sum(d< -1e-9 for d in delta)]
    lines = ['# RBOT数据集五方法评测结果汇总', '',
             '日期：2026-10-23。依据本次提供的 `summary.json` 和 `results.md`；输入原件已另存并计算SHA256，没有覆盖之前两个工业数据集的结果。', '',
             '**我们的方法成功率为 %.4f%%，相对ICG+提高 %.4f 个百分点，多成功 %s 帧；计算均值 %.3f ms/帧，计算FPS %.2f。**' %
             (ours['success_percent'], difference, format(ours['successes']-base['successes'], ','), ours['timing']['compute_wall_us']['mean_us']/1000, ours['derived_compute_fps']), '',
             '## 1. 总体准确率', '',
             table(['方法', '评价帧', '成功帧', '失败并重置帧', '成功率', '位姿重算成功率', '相对ICG+ / 百分点'],
                   [[LABELS[k], format(m[k]['frames'], ','), format(m[k]['successes'], ','), format(m[k]['reset_count'], ','),
                     '%.4f%%'%m[k]['success_percent'], '%.4f%%'%m[k]['recomputed_success_percent'],
                     '%+.4f'%(m[k]['success_percent']-base['success_percent'])] for k in METHODS]), '',
             '相对ICG+，72个配置中我们有 **%d个提高、%d个持平、%d个下降**。失败并重置帧从%s降至%s，减少 **%.2f%%**；这是本协议下的失败计数，不能解释为无真值干预的连续运行可靠性。' %
             (*wtl, format(base['reset_count'], ','), format(ours['reset_count'], ','), 100*(base['reset_count']-ours['reset_count'])/base['reset_count']), '',
             '## 2. 范围、时间和输入协议', '',
             '- 全部18个物体×4个序列＝72个目标配置；5方法共360个作业，完整配对。',
             '- 原生640×512 RGB；图像0000仅用于目标真值初始化，0001–1000用于评价；每方法72,000帧。',
             '- 平移误差≤5 cm且旋转误差≤5°成功；失败帧先计入，再按该帧真值重置；评估使用重置前估计。',
             '- 不输入其他物体真值或已知遮挡物模型，不包括额外的modeled-occlusion第五配置。',
             '- ICG表示原版RGB/Region入口；ICG+表示原版Region+ORB入口，JSON内部ID保留icg_rgb和m3t_original_rgb；不把它们写成RGB-D测试。',
             '- 每配置前10个评价帧仅从计时剔除，准确率仍统计；每方法计时71,280帧。单轮运行，固定单线程、软件OpenGL llvmpipe，DISK使用选定GPU。',
             '- 主速度为 `compute_wall_us = tracking_wall_us - decode_us`，包含预测、图像转换/噪声处理、渲染、特征、匹配和优化；不包含读图/PNG解码、误差评价、GT重置、初始化、模型生成、会话加载或写结果。', '',
             '## 3. 四序列准确率', '',
             table(['序列', '评价帧/方法']+[LABELS[k] for k in METHODS]+['我们相对ICG+ / 百分点'],
                   [[s+' / '+n, '18,000']+['%.4f%%'%m[k]['by_sequence'][s]['success_percent'] for k in METHODS]+
                    ['%+.4f'%(m['ours']['by_sequence'][s]['success_percent']-base['by_sequence'][s]['success_percent'])]
                    for s,n in zip(SEQUENCES, SEQUENCE_NAMES)]), '',
             '最大增益来自c_noisy：+8.1389个百分点，多成功1,465帧。该观察只能说明整套方法在此子集受益较大；尚不能把增益全部归给去噪、DISK或时序模块，需同一代码基底的消融。', '',
             '## 4. 18个物体的四序列平均', '',
             table(['物体']+[LABELS[k] for k in METHODS]+['我们相对ICG+ / 百分点'],
                   [[b]+['%.4f%%'%m[k]['by_body'][b]['success_percent'] for k in METHODS]+
                    ['%+.4f'%(m['ours']['by_body'][b]['success_percent']-base['by_body'][b]['success_percent'])] for b in bodies]), '',
             '四序列合并后，18个物体上我们均高于ICG+。glue仍只有83.9750%，是当前主要难点之一；“每个物体平均更好”不代表每个单独配置都最好。', '',
             '## 5. 每帧计算耗时', '',
             table(['方法', '计时帧', '均值 ms', '中位数 ms', 'P95 ms', '计算FPS'],
                   [[LABELS[k], '71,280']+['%.3f'%(m[k]['timing']['compute_wall_us'][f]/1000) for f in ['mean_us','median_us','p95_us']]+
                    ['%.2f'%m[k]['derived_compute_fps']] for k in METHODS]), '',
             '我们的均值为ICG+的 **%.3f倍**。平均31.25计算FPS不能等价为稳定30 Hz实时系统：本轮计算P95为58.767 ms，且这里不包含输入读取、输出显示和失败重置。速度是本次共享服务器负载下的单轮实测，论文正式计时应另做固定条件重复。' %
             (ours['timing']['compute_wall_us']['mean_us']/base['timing']['compute_wall_us']['mean_us']), '',
             '原生cycle的阶段划分按方法不同，只用于定位各自热点；不取代上表统一计算耗时。', '',
             table(['阶段', 'ICG+ ms', '我们 ms'], [[label, '%.3f'%(base['timing'][key]['mean_us']/1000),
                    '%.3f'%(ours['timing'][key]['mean_us']/1000)] for key,label in
                    [('correspondences_us','对应关系'),('gradient_hessian_us','梯度/Hessian'),('optimization_us','优化'),('results_us','结果/关键帧维护')]]), '',
             '我们对应关系阶段约24.198 ms，占原生cycle约75.72%；其中包含CAD渲染、图像读回、DISK提取、匹配与RANSAC等，现有统计不能进一步断言全部时间都消耗在神经网络。预处理均值约0.383 ms。', '',
             '## 6. 审计、例外和旧结果对照', '',
             '所有方法无效估计均为0。360个作业的帧数、成功/失败计数、分组聚合、配对表和源Markdown的360个配置成功率单元格均已交叉核验。没有收到逐帧CSV，因此本次核验不能代替服务器逐帧位姿重算。', '',
             table(['方法/配置', '图像索引', '原生成功率', '重算成功率'],
                   [[LABELS[r['method']]+' / '+r['id'], ', '.join(str(i) for i in r['success_classification_disagreement_images']),
                     '%.4f%%'%r['success_percent'], '%.4f%%'%r['recomputed_success_percent']] for r in discrepancies]), '',
             'SRT3D有1帧的原生与重算分类不同，总体分别为90.9097%和90.9111%。主表保留控制实际重置轨迹的原生口径；这1帧差异的具体数值原因仍需对应CSV，不能直接推定为阈值边界误差。', '',
             '我们不优于ICG+的两项为b_dynamiclight/cat（99.9%对100.0%）与c_noisy/ape（97.3%对98.2%）。此外c_noisy/cat、d_occlusion/cat、d_occlusion/eggbox、d_occlusion/iron也低于其余方法中的最佳值，应保留为失败分析。', '',
             '旧轮我们为97.0708%，本轮97.0292%，相差-0.0417个百分点（对应相同分母少30个成功帧）。旧轮没有统一计算计时，新轮关闭额外诊断并使用不同评测壳/有限数值保护；这不是独立消融，不能将差值归因给某个算法模块，也不替换本轮完整配对结果。', '',
             '按物体聚类、四序列共同重采样的描述性bootstrap：增益均值%+.4f个百分点，95%%百分位区间[%+.4f, %+.4f]；18个聚类、20,000次、种子20261023。该区间反映本组物体抽样差异，不使用72,000帧作为独立样本，也不是新工位泛化保证。' %
             (bootstrap['mean'], *bootstrap['percentile_95_interval']), '',
             '## 7. 与已完成两套工业数据的结果并列', '',
             table(['数据集', '配置/方法']+[LABELS[k] for k in METHODS]+['我们相对ICG+ / 百分点'],
                   [[name, str(len(v['paired_configurations']))]+['%.4f%%'%v['methods'][k]['success_percent'] for k in METHODS]+
                    ['%+.4f'%(v['methods']['ours']['success_percent']-v['methods']['m3t_original_rgb']['success_percent'])]
                    for name,v in [('RBOT',value),('工业基础集（上一轮完整集）',industrial),('工业进阶集v1.0',advanced)]]), '',
             table(['数据集']+[LABELS[k]+' 计算ms' for k in METHODS],
                   [[name]+['%.3f'%(v['methods'][k]['timing']['compute_wall_us']['mean_us']/1000) for k in METHODS]
                    for name,v in [('RBOT',value),('工业基础集',industrial),('工业进阶集',advanced)]]), '',
             '三套数据分别支持“公共基准、工业外观/运动、受控困难条件”的验证。分辨率、对象和场景复杂度不同，服务器运行负载也不同，不拼成一个总成功率或用跨数据集时间变化证明加速。两套自渲染集和RBOT仍不足以替代真实工件下游实验。', '',
             '## 8. 图与运行身份', '',
             '![四序列准确率及计算开销对比](figures/'+figure.name+'.png)', '',
             '图中速度使用统一计算耗时，右图横轴为对数轴；准确率为单轮完整GT-reset协议结果。SVG同名文件可用于后续排版。', '',
             '- 服务器运行目录：`'+str(Path(value['configurations'][0]['pose_csv']).as_posix()).split('/configurations/')[0]+'`。',
             '- GPU UUID：`'+value['identity']['selected_gpu']+'`。',
             '- 输入指纹：`'+value['identity']['inputs']['input_fingerprint']+'`。',
             '- 原件与本地复核记录：`'+str(archive)+'`。', '',
             '## 附录：18物体×4序列，五方法全部配对', '',
             table(['物体', '序列']+[LABELS[k] for k in METHODS]+['我们相对ICG+ / 百分点'],
                   [[b,s]+['%.4f%%'%next(p for p in paired if p['body']==b and p['sequence']==s)['success_percent'][k] for k in METHODS]+
                    ['%+.4f'%(next(p for p in paired if p['body']==b and p['sequence']==s)['success_percent']['ours']-
                               next(p for p in paired if p['body']==b and p['sequence']==s)['success_percent']['m3t_original_rgb'])]
                    for b in bodies for s in SEQUENCES]), '']
    return '\n'.join(lines), wtl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'source-report', 'industrial', 'advanced', 'output', 'archive']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    raw = args.input.read_bytes(); md_raw = args.source_report.read_bytes()
    value = json.loads(raw.decode('utf-8-sig'))
    bodies, discrepancies = validate(value, md_raw.decode('utf-8-sig'))
    industrial = json.loads(args.industrial.read_text(encoding='utf-8-sig'))
    advanced = json.loads(args.advanced.read_text(encoding='utf-8-sig'))
    if industrial['status'] != 'complete' or advanced['status'] != 'complete':
        raise ValueError('Incomplete historical comparison inputs')
    args.archive.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.original.json',raw), ('results.original.md',md_raw)]:
        target = args.archive/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Archive already contains different source data: '+str(target))
        target.write_bytes(data)
    bootstrap = object_bootstrap(value)
    figure = args.output.parent/'figures/rbot-five-methods-20261023'
    plot(value, figure)
    report, wtl = create_report(value, bodies, discrepancies, bootstrap, args.archive, industrial, advanced, figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding='utf-8')
    audit = dict(status='passed_exported_result_consistency', source_summary_sha256=digest(raw),
                 source_report_sha256=digest(md_raw), configurations=72, jobs=360,
                 source_markdown_configuration_numeric_cells_checked=360,
                 evaluated_frames_per_method=72000, timed_frames_per_method=71280,
                 primary_timing_metric='compute_wall_us', paired_wins_ties_losses_against_icg_plus=wtl,
                 discrepancies=discrepancies, object_cluster_bootstrap=bootstrap,
                 report=str(args.output), report_sha256=digest(args.output.read_bytes()),
                 historical_sources={str(path):digest(path.read_bytes()) for path in [args.industrial,args.advanced]},
                 limit='Local exported JSON/Markdown consistency only; no server CSVs reprocessed')
    (args.archive/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['discrepancies','historical_sources']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
