"""Reformat saved five-method summaries using computation-only timing; no inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path

LABELS = {'rbgt': 'RBGT', 'srt3d': 'SRT3D', 'icg_rgb': 'ICG', 'm3t_original_rgb': 'ICG+', 'ours': 'M3T (ours final)'}


def timing_table(summary):
    methods = summary['identity']['methods']
    lines = ['| 方法 | 成功率 | 每帧计算均值 ms | 计算中位数 ms | 计算P95 ms | 计算FPS |',
             '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        item = summary['methods'][m]
        if 'compute_wall_us' not in item.get('timing', {}):
            raise ValueError('Cannot reconstruct compute time from cycle-only results; rerun required')
        t = item['timing']['compute_wall_us']
        if any(not math.isfinite(t[k]) or t[k] <= 0 for k in ['mean_us', 'median_us', 'p95_us']):
            raise ValueError('Invalid computation timing')
        lines.append('| %s | %.4f%% | %.3f | %.3f | %.3f | %.2f |' %
                     (LABELS.get(m, m), item['success_percent'], t['mean_us']/1000,
                      t['median_us']/1000, t['p95_us']/1000, 1e6/t['mean_us']))
    return '\n'.join(lines)+'\n'


def format_report(summary):
    if summary['status'] != 'complete':
        raise ValueError('Only complete reports can be reformatted')
    methods = summary['identity']['methods']
    lines = ['# 五方法结果：每帧计算耗时口径', '',
             '范围：%s。成功率沿用原结果；本次只改变时间展示口径，不重新推理。' % summary.get('scope', summary.get('dataset', '')), '',
             '**主要计时排除读图和PNG解码**，保留预测、算法预处理和跟踪；不含误差评价、GT重置、初始化、模型生成、输出写入。', '',
             timing_table(summary), '',
             '计算FPS为平均计算耗时的倒数，不等同包含读图的在线吞吐；预热仍沿用原记录，准确率分母不变。共享资源负载的影响仍保留。', '']
    for key in ['sequence', 'condition', 'scene', 'object', 'body']:
        groups = summary['methods'][methods[0]].get('by_'+key)
        if not groups:
            continue
        lines += ['## 按%s统计成功率' % key, '', '| 分组 | '+' | '.join(LABELS.get(m, m) for m in methods)+' |', '|---|'+'---:|'*len(methods)]
        for name in sorted(groups):
            lines.append('| '+name+' | '+' | '.join('%.4f%%' % summary['methods'][m]['by_'+key][name]['success_percent'] for m in methods)+' |')
        lines.append('')
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve():
        p.error('Choose a separate Markdown output; preserve the original summary')
    raw = args.input.read_bytes()
    doc = format_report(json.loads(raw.decode('utf-8-sig')))
    doc += '\n来源：`%s`；SHA256：`%s`。\n' % (args.input, hashlib.sha256(raw).hexdigest())
    args.output.write_text(doc, encoding='utf-8')
    if args.input.read_bytes() != raw:
        raise RuntimeError('Input summary changed during formatting')
    print(str(args.output))


if __name__ == '__main__':
    main()
