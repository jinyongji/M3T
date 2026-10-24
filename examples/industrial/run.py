#!/usr/bin/env python3
"""Sequential, resumable final-algorithm evaluation, one GPU and one Xvfb."""
import argparse
import contextlib
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dataset import (CONDITIONS, MOTIONS, DISK_SHA256, configurations, preflight,
                     sha256, write_json)
from metrics import aggregate, analyze


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def summarize(output, results, scope):
    report = {"schema_version": 1, "scope": scope, "algorithm": "current_final",
              "protocol": "GT initialization at image 1; images 2..N; failed frame counted then GT reset",
              "translation_threshold_m": .05, "rotation_threshold_deg": 5,
              "rotation_method": "native float clamped angle; audit uses double polar projection",
              "success_percent_definition": "native labels on the actual reset trajectory; audit is not a rerun",
              "invalid_estimate_policy": "count as failure; never drop from the success denominator",
              "timing_scope": "M3T measured cycle incl preprocessing; excludes image decoding and setup/cache generation",
              "overall": aggregate(results), "by_condition": {}, "by_motion": {}, "by_object": {}, "by_scene": {},
              "configurations": results}
    for key in ("condition", "motion", "object", "scene"):
        report["by_" + key] = {name: aggregate([c for c in results if c[key] == name])
                              for name in sorted({c[key] for c in results})}
    write_json(output / "summary.json", report)
    write_json(output / "pose_metrics_5cm_5deg.json", report)
    columns = ["scene", "object", "motion", "condition", "frames", "success_percent",
               "recomputed_success_percent", "reset_count", "invalid_estimate_count", "cycle_mean_us"]
    with (output / "success_rates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["# 工业数据集 M3T 评测", "", "范围：%s；配置 %d；评测帧 %d。" %
             (scope, len(results), report["overall"]["frames"]), "",
             "当前 final 配置，5 cm / 5°；首帧真值初始化、失败后重置，遮挡帧保留。成功率按目标帧计数。", "",
             "原生成功率 %.4f%%；位姿重算 %.4f%%；无效估计 %d；两者判定差异 %d 帧。" %
             (report["overall"]["success_percent"], report["overall"]["recomputed_success_percent"],
              report["overall"]["invalid_estimate_count"], report["overall"]["success_classification_disagreement_count"]), ""]
    for motion in MOTIONS:
        lines += ["## " + motion, "", "| 目标 | " + " | ".join(CONDITIONS) + " |",
                  "|---|" + "---:|"*len(CONDITIONS)]
        for obj in sorted({c["object"] for c in results if c["motion"] == motion}):
            row = []
            for condition in CONDITIONS:
                found = [c for c in results if c["object"] == obj and c["motion"] == motion and c["condition"] == condition]
                row.append("%.2f%%" % found[0]["success_percent"] if found else "—")
            lines.append("| %s | %s |" % (obj, " | ".join(row)))
        lines.append("")
    lines += ["时间不含图像读取、模型预计算和进程启动；OpenGL 使用服务器当前 Mesa/llvmpipe，DISK 使用 CUDA。",
              "两个 JSON 均保留每个配置的原生与重算成功率、重置次数和无效估计数。位姿误差均值仅使用有效估计，分母另行记录。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _main(resources):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/jyj/Projects/M3T"))
    parser.add_argument("--dataset", type=Path, default=Path("/home/jyj/autodl-tmp/industrial-render-900x600/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016"))
    parser.add_argument("--cache", type=Path, default=Path("/home/jyj/autodl-tmp/M3T-industrial-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=500, help="Includes first GT initialization image")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--motion", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--object", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    # Exclusive file lock survives SSH disconnection and is released even if a
    # process is killed. Do not mistake stale PIDs for an active evaluation.
    import fcntl
    lock = resources.enter_context((output / ".run.lock").open("a"))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This output directory already has an active evaluation")
    old = None
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("Output exists; choose a new directory or use resume")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif args.resume:
        raise SystemExit("No run manifest to resume")
    configs = configurations(args.scene, args.motion, args.condition, args.object)
    binary = args.source / "build-linux/examples/evaluate_industrial_dataset"
    model = args.source / "disk_512.onnx"
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("CUDA_VISIBLE_DEVICES must identify your allocated GPU")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit("Build evaluate_industrial_dataset first")
    if sha256(model) != DISK_SHA256:
        raise SystemExit("DISK model SHA256 does not match the RBOT model")
    if not shutil.which("xvfb-run"):
        raise SystemExit("Private xvfb-run is missing; source the existing user environment")
    implementation = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    scope = "complete dataset" if len(configs) == 150 and args.frames == 500 else "selected subset / smoke test"
    identity = {"dataset": str(args.dataset.resolve()), "binary_sha256": sha256(binary),
                "model_sha256": DISK_SHA256, "implementation_sha256": implementation,
                "frames_including_initialization": args.frames, "configurations": configs,
                "algorithm": "current_final", "scope": scope}
    if old and old["identity"] != identity:
        raise SystemExit("Resume identity differs (binary, scripts, model, dataset or selection changed)")
    manifest = {"schema_version": 1, "status": "validating", "pid": os.getpid(), "identity": identity,
                "started_at": old["started_at"] if old else utc(), "updated_at": utc(),
                "expected_configurations": len(configs), "expected_evaluated_frames": len(configs)*(args.frames-1),
                "completed_configurations": list(old.get("completed_configurations", [])) if old else [],
                "output": str(output), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    if old and "dataset_fingerprint" in old:
        manifest["dataset_fingerprint"] = old["dataset_fingerprint"]
    write_json(manifest_path, manifest)
    began = time.monotonic()
    try:
        print("Validating selected poses, CAD diffuse textures and PNG headers...", flush=True)
        validation = preflight(args.dataset, configs, args.frames)
        if old and old.get("dataset_fingerprint") != validation["fingerprint"]:
            raise ValueError("Dataset fingerprint changed; start a separate run")
        write_json(output / "input_validation.json", validation)
        manifest["dataset_fingerprint"] = validation["fingerprint"]
        cache = args.cache / validation["fingerprint"][:16]
        cache.mkdir(parents=True, exist_ok=True)
        manifest["cache"] = str(cache)
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        for command, filename in ((["nvidia-smi"], "gpu.txt"),
                                  (["git", "-C", str(args.source), "rev-parse", "HEAD"], "git_head.txt"),
                                  (["git", "-C", str(args.source), "diff", "--stat"], "git_diff_stat.txt")):
            with (output / filename).open("w", encoding="utf-8") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        results = []
        completed = set(old.get("completed_configurations", [])) if old else set()
        for number, config in enumerate(configs, 1):
            directory = output / "configurations" / config["id"]
            directory.mkdir(parents=True, exist_ok=True)
            manifest.update(status="running", current=config, job=number, updated_at=utc())
            write_json(manifest_path, manifest)
            if config["id"] not in completed:
                command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", str(binary),
                           "--dataset", str(args.dataset), "--cache", str(cache), "--output", str(directory),
                           "--disk-model", str(model), "--frames", str(args.frames)]
                for key in ("scene", "motion", "condition", "object"):
                    command += ["--" + key, config[key]]
                write_json(directory / "command.json", command)
                print("[%d/%d] %s" % (number, len(configs), config["id"]), flush=True)
                with (directory / "evaluation.log").open("a", encoding="utf-8") as stream:
                    stream.write("\n=== attempt %s ===\n" % utc())
                    stream.flush()
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            metrics = analyze(directory, args.dataset, config, args.frames)
            write_json(directory / "metrics.json", metrics)
            results.append(metrics)
            if config["id"] not in manifest["completed_configurations"]:
                manifest["completed_configurations"].append(config["id"])
            manifest["completed_evaluated_frames"] = sum(c["frames"] for c in results)
            manifest["updated_at"] = utc()
            write_json(manifest_path, manifest)
            print("  success %.3f%%; audit %.3f%%; invalid %d" %
                  (metrics["success_percent"], metrics["recomputed_success_percent"], metrics["invalid_estimate_count"]), flush=True)
        report = summarize(output, results, scope)
        if report["overall"]["frames"] != manifest["expected_evaluated_frames"]:
            raise ValueError("Final denominator differs from the requested protocol")
        manifest.update(status="complete", updated_at=utc(), elapsed_this_attempt_s=time.monotonic()-began)
        write_json(manifest_path, manifest)
        print("Complete: %s\nSuccess: %.4f%%" % (output, report["overall"]["success_percent"]), flush=True)
    except BaseException as error:
        manifest.update(status="failed", updated_at=utc(), error=str(error))
        write_json(manifest_path, manifest)
        raise


def main():
    with contextlib.ExitStack() as resources:
        _main(resources)


if __name__ == "__main__":
    main()
