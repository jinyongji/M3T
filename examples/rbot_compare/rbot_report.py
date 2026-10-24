"""Paired RBOT reports with decode-free computation as the primary speed metric."""
import csv
import math
from pathlib import Path
import statistics
from rbot_data import LABELS, write_json
from rbot_metrics import TIMES, timing_rows


def aggregate(rows):
    frames = sum(r['frames'] for r in rows)
    if not frames:
        raise ValueError('Empty aggregation')
    return dict(configurations=len(rows), frames=frames, successes=sum(r['successes'] for r in rows),
                reset_count=sum(r['reset_count'] for r in rows),
                success_percent=100*sum(r['successes'] for r in rows)/frames,
                recomputed_success_percent=100*sum(r['recomputed_successes'] for r in rows)/frames,
                invalid_estimate_count=sum(r['invalid_estimate_count'] for r in rows),
                success_classification_disagreement_count=sum(r['success_classification_disagreement_count'] for r in rows))


def describe(values):
    values = sorted(values)
    return dict(mean_us=statistics.mean(values), median_us=statistics.median(values),
                p95_us=values[max(0, math.ceil(.95*len(values))-1)], frames=len(values))


def summarize(output, results, identity, legacy=None):
    output = Path(output)
    methods, configs = identity['methods'], identity['configurations']
    evaluated_images = identity.get('frames', max(r['frames'] for r in results)+1)-1 if results else 0
    expected_keys = {(c['id'], m) for c in configs for m in methods}
    if len(results) != len(expected_keys) or {(r['id'], r['method']) for r in results} != expected_keys:
        raise ValueError('Incomplete/unpaired five-method results')
    grouped = {}
    for m in methods:
        items = [r for r in results if r['method'] == m]
        times = [t for r in items for t in timing_rows(r['pose_csv'], identity['warmup'])]
        grouped[m] = aggregate(items)
        grouped[m]['timing'] = {k: describe([t[k] for t in times]) for k in TIMES}
        grouped[m]['derived_compute_fps'] = 1e6/grouped[m]['timing']['compute_wall_us']['mean_us']
        for key in ['sequence', 'body']:
            grouped[m]['by_'+key] = {name: aggregate([r for r in items if r[key] == name]) for name in sorted({r[key] for r in items})}
    paired = [dict(c, success_percent={m: next(r['success_percent'] for r in results if r['id'] == c['id'] and r['method'] == m) for m in methods}) for c in configs]
    report = dict(schema_version=1, status='complete', dataset='RBOT', methods=grouped,
                  thresholds=dict(translation_m=.05, rotation_deg=5), identity=identity,
                  scope='RBOT: %d target configurations, %d methods, %d evaluated frames/configuration' % (len(configs), len(methods), evaluated_images),
                  protocol='RGB; GT image 0000 excluded; evaluate images 0001..%04d; count failure then reset to its GT; no modeled occluder' % evaluated_images,
                  primary_timing_metric='compute_wall_us',
                  primary_timing_scope='prediction, image preprocessing and tracking; excludes file read/PNG decode, metrics, GT reset, setup and serialization',
                  timing_warmup_excluded_per_configuration=identity['warmup'],
                  native_cycle_note='method-specific diagnostic only; not used as cross-method speed',
                  configurations=results, paired_configurations=paired, previous_ours_run=legacy)
    write_json(output/'summary.json', report)
    lines = ['# RBOT%s评测' % ('五方法' if len(methods) == 5 else '方法'), '', '状态：complete；%d个目标配置，%d个方法作业；每配置%d个评价帧。' % (len(configs), len(results), evaluated_images), '',
             'RGB；首帧0000真值初始化；失败帧计入后真值重置；5cm/5°；本次四序列不使用已知遮挡物模型。', '',
             '**主要速度指标：每帧计算耗时，排除读图与PNG解码。** 包含算法图像预处理、预测和跟踪；排除评价、GT重置、初始化、模型生成、CSV写入。', '',
             '| 方法 | 目标帧数 | 成功率 | 重算成功率 | 无效估计 | 计算均值 ms | 计算中位数 ms | 计算P95 ms | 计算FPS |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        x = grouped[m]; t = x['timing']['compute_wall_us']
        lines.append('| %s | %d | %.4f%% | %.4f%% | %d | %.3f | %.3f | %.3f | %.2f |' %
                     (LABELS[m], x['frames'], x['success_percent'], x['recomputed_success_percent'], x['invalid_estimate_count'],
                      t['mean_us']/1000, t['median_us']/1000, t['p95_us']/1000, x['derived_compute_fps']))
    if 'ours' in grouped and 'm3t_original_rgb' in grouped:
        delta = grouped['ours']['success_percent']-grouped['m3t_original_rgb']['success_percent']
        lines += ['', '我们的总体成功率相对ICG+：%+.4f个百分点。' % delta]
    for key in ['sequence', 'body']:
        lines += ['', '## 按%s统计' % ('序列' if key == 'sequence' else '物体'), '',
                  '| 分组 | '+' | '.join(LABELS[m] for m in methods)+' |', '|---|'+'---:|'*len(methods)]
        for name in sorted({r[key] for r in results}):
            lines.append('| '+name+' | '+' | '.join('%.4f%%' % grouped[m]['by_'+key][name]['success_percent'] for m in methods)+' |')
    lines += ['', '## 全部配置', '', '| 物体 | 序列 | '+' | '.join(LABELS[m] for m in methods)+' |', '|---|---|'+'---:|'*len(methods)]
    for p in paired:
        lines.append('| '+p['body']+' | '+p['sequence']+' | '+' | '.join('%.4f%%' % p['success_percent'][m] for m in methods)+' |')
    lines += ['', '速度为本次共享服务器负载下的单轮实测。计算FPS是平均计算耗时的倒数，不表示包含读图的在线吞吐。预热仅从计时剔除，不从准确率剔除。',
              '逐帧读图、完整流程与原生cycle字段只供JSON/CSV诊断，不作为主表速度。主准确率采用控制真实重置轨迹的原生判定；位姿重算结果独立列出。', '']
    if legacy:
        lines += ['历史改进版运行仅保留作准确率参考，未拼入本轮五方法速度或配对分母：`'+legacy['path']+'`。', '']
        if 'historical_native_success_percent' in legacy:
            lines += ['旧轮改进版总体成功率为%.4f%%。本轮使用统一计时、关闭额外诊断，并采用有限误差/无效估计保护；新旧差值需结合运行身份解释，不强制结果相等。' % legacy['historical_native_success_percent'], '']
    (output/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'success_rates.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['method', 'body', 'sequence', 'frames', 'success_percent', 'recomputed_success_percent', 'reset_count', 'invalid_estimate_count']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
    return report
