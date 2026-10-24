"""Timing audits and paired reports. All durations are microseconds."""
import csv
import math
import statistics
from pathlib import Path

from dataset import write_json
from metrics import aggregate

METHODS = ["rbgt", "srt3d", "icg_rgb", "m3t_original_rgb", "ours"]
LABELS = {"rbgt": "RBGT", "srt3d": "SRT3D", "icg_rgb": "ICG",
          "m3t_original_rgb": "ICG+", "ours": "M3T (ours final)"}
TIMES = ["cycle_us", "decode_us", "preprocessing_us", "tracking_wall_us", "compute_wall_us",
         "metric_us", "reset_us", "frame_wall_us", "before_camera_us", "correspondences_us",
         "gradient_hessian_us", "optimization_us", "results_us"]


def timing_rows(csv_file, warmup):
    with Path(csv_file).open(encoding="utf-8-sig", newline="") as stream:
        rows = [{k: float(row[k]) for k in TIMES} for row in csv.DictReader(stream)]
    if not 0 <= warmup < len(rows):
        raise ValueError("Warmup must leave at least one timed frame")
    for row in rows:
        if any(not math.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("Nonfinite/negative timing")
        close = lambda a, b: abs(a-b) <= max(2, abs(a)*1e-4)
        if not close(row["compute_wall_us"], row["tracking_wall_us"]-row["decode_us"]):
            raise ValueError("Decode/tracking timing decomposition disagrees")
        if not close(row["frame_wall_us"], row["tracking_wall_us"]+row["metric_us"]+row["reset_us"]):
            raise ValueError("Frame timing decomposition disagrees")
        if row["preprocessing_us"] > row["compute_wall_us"]+2:
            raise ValueError("Preprocessing exceeds compute wall time")
    return rows[warmup:]


def describe(values):
    ordered = sorted(values)
    return {"mean_us": statistics.mean(ordered), "median_us": statistics.median(ordered),
            "p95_us": ordered[max(0, math.ceil(.95*len(ordered))-1)], "frames": len(ordered)}


def summarize(output, results, warmup, identity, scope, partial=False):
    output = Path(output)
    methods = identity["methods"]
    grouped = {}
    for method in methods:
        items = [r for r in results if r["method"] == method]
        if not items:
            continue
        times = [row for r in items for row in timing_rows(r["pose_csv"], warmup)]
        overall = aggregate(items)
        overall["timing"] = {key: describe([r[key] for r in times]) for key in TIMES}
        overall["derived_compute_fps"] = 1e6/overall["timing"]["compute_wall_us"]["mean_us"]
        overall["derived_frame_loop_fps"] = 1e6/overall["timing"]["frame_wall_us"]["mean_us"]
        overall["repeats"] = identity["repeats"]
        overall["by_repeat"] = {str(i): aggregate([r for r in items if r["repeat"] == i])
                                for i in sorted({r["repeat"] for r in items})}
        for key in ("condition", "motion", "object", "scene"):
            overall["by_"+key] = {name: aggregate([r for r in items if r[key] == name])
                                  for name in sorted({r[key] for r in items})}
        grouped[method] = overall
    paired = []
    for config in identity["configurations"]:
        for repeat in range(1, identity["repeats"]+1):
            items = {r["method"]: r for r in results if r["id"] == config["id"] and r["repeat"] == repeat}
            if set(items) != set(methods):
                if partial:
                    continue
                raise ValueError("Unpaired configuration: "+config["id"])
            if len({r["frames"] for r in items.values()}) != 1:
                raise ValueError("Methods used different denominators")
            paired.append(dict(config, repeat=repeat,
                               success_percent={k: v["success_percent"] for k, v in items.items()}))
    comparisons = {}
    if not partial and "m3t_original_rgb" in grouped:
        original = grouped["m3t_original_rgb"]
        for method, item in grouped.items():
            if method == "m3t_original_rgb":
                continue
            comparisons[method] = {"success_delta_percentage_points": item["success_percent"]-original["success_percent"],
                "compute_speed_ratio_original_over_method": original["timing"]["compute_wall_us"]["mean_us"]/item["timing"]["compute_wall_us"]["mean_us"],
                "frame_loop_speed_ratio_original_over_method": original["timing"]["frame_wall_us"]["mean_us"]/item["timing"]["frame_wall_us"]["mean_us"]}
    report = {"schema_version": 1, "status": "partial" if partial else "complete", "scope": scope,
              "protocol": "RGB only; GT image 1 excluded; evaluate 2..N; count failed frame then reset to its GT",
              "thresholds": {"translation_m": .05, "rotation_deg": 5},
              "success_labels": "native float, audit double polar; audit does not change reset trajectory",
              "invalid_estimates": "count as failures; keep in denominator",
              "timing_warmup_excluded_per_configuration": warmup,
              "primary_timing_metric": "compute_wall_us",
              "primary_timing_scope": "per-frame prediction, preprocessing and tracking; excludes file read/PNG decode, metric calculation, GT reset, setup and serialization",
              "timing_scope": "frame_wall includes PNG decode, conversion/filtering, prediction, tracking, metrics and GT reset; excludes setup, overlay, CSV serialization and process startup",
              "timing_note": "report compute_wall as primary; frame_wall and decode are diagnostics; native cycle scopes differ between algorithms; all-frame accuracy includes timing warmup",
              "input_fingerprint_scope": "metadata, models and poses SHA256; PNG IHDR hash, size and mtime, not full pixel hashes",
              "identity": identity, "methods": grouped, "paired_configurations": paired,
              "comparisons_to_original_m3t": comparisons,
              "configurations": results}
    suffix = "partial_" if partial else ""
    write_json(output/(suffix+"summary.json"), report)
    columns = ["method", "repeat", "scene", "motion", "condition", "object", "frames", "success_percent",
               "recomputed_success_percent", "reset_count", "invalid_estimate_count", "cycle_mean_us"]
    with (output/(suffix+"success_rates.csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    lines = ["# 工业数据集算法对比", "", "范围：%s；状态：%s。" % (scope, report["status"]), "",
             "RGB 输入；首帧 GT 初始化；失败帧计入后重置；5 cm / 5°。预热只从时间统计排除。", "",
             "| 方法 | 目标帧数（含重复） | 成功率 | 重算成功率 | 无效估计 | 每帧计算均值 ms | 计算 P95 ms | 计算 FPS |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, item in grouped.items():
        lines.append("| %s | %d | %.4f%% | %.4f%% | %d | %.3f | %.3f | %.3f |" %
                     (LABELS[method], item["frames"], item["success_percent"], item["recomputed_success_percent"],
                      item["invalid_estimate_count"], item["timing"]["compute_wall_us"]["mean_us"]/1000,
                      item["timing"]["compute_wall_us"]["p95_us"]/1000, item["derived_compute_fps"]))
    for key in ("motion", "condition", "object"):
        lines += ["", "## 按 "+key+" 的成功率", "", "| 分组 | "+" | ".join(LABELS[m] for m in grouped)+" |",
                  "|---|"+"---:|"*len(grouped)]
        names = sorted({r[key] for r in results})
        for name in names:
            lines.append("| %s | %s |" % (name, " | ".join("%.3f%%" % grouped[m]["by_"+key][name]["success_percent"]
                                                          if name in grouped[m]["by_"+key] else "—" for m in grouped)))
    lines += ["", "主要速度指标只显示每帧计算耗时：包含预测、转换/算法预处理和跟踪，排除文件读取、PNG解码、误差统计及GT重置。计算FPS为计算均值的倒数，不代表包含读图的在线吞吐。",
              "完整帧耗时与读图解码耗时仅保留在JSON中供诊断，不作为主要速度结果。",
              "时间排除首帧初始化、模型生成、ONNX 会话加载、投影图输出、CSV 写入和进程启动。JSON 保留分阶段均值、中位数、P95，以及每个配置和每轮的成功率。",
              "ICG 使用原仓库 RBOT 的 RGB/Region 配置；ICG+ 使用原版 Region+ORB 配置；本次均为 RGB 输入。原生 cycle 仅作内部诊断，不跨算法直接比较。", ""]
    (output/(suffix+"results.md")).write_text("\n".join(lines), encoding="utf-8")
    return report
