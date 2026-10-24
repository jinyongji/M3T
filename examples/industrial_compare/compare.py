#!/usr/bin/env python3
"""Sequential, paired industrial tracking benchmark; no sudo or Python packages."""
import argparse
import contextlib
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from provenance import SHARED, verified_build
from dataset import (DISK_SHA256, configurations, preflight, read_camera, sha256, write_json)
from metrics import analyze
from report import METHODS, summarize, timing_rows

KIT = Path(__file__).resolve().parent
DEFAULT_DATA = "/home/jyj/autodl-tmp/industrial-render-900x600/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def jobs(configs, methods, repeats):
    # Rotate the first method to distribute warm OS image-cache advantage.
    for repeat in range(1, repeats+1):
        for index, config in enumerate(configs):
            offset = (index+repeat-1) % len(methods)
            for method in methods[offset:]+methods[:offset]:
                yield {"key": "%02d/%s/%s" % (repeat, config["id"], method),
                       "method": method, "repeat": repeat, "config": config}


def hold_lock(resources, path, shared=False, existing_only=False):
    import fcntl
    path = Path(path)
    if existing_only and not path.is_file():
        return
    stream = resources.enter_context(path.open("rb" if existing_only else "a"))
    try:
        fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("Active task holds %s; wait for it to finish before benchmarking" % path)


def protect_dataset(resources, dataset):
    # Same flock files as the rendering/publishing controllers. A shared lock
    # allows reads but prevents the known controllers from replacing data.
    runtime = Path(dataset).resolve().parent.parent/"中间结果文件"
    for state in ("run_state", "run_state_v09"):
        hold_lock(resources, runtime/state/"render.lock", shared=True, existing_only=True)


def command_for(job, args, cache, directory):
    command = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24",
               str(KIT/"build"/job["method"]/"evaluate_industrial_baseline"),
               "--dataset", str(args.dataset), "--cache", str(cache), "--output", str(directory),
               "--frames", str(args.frames)]
    for key in ("scene", "motion", "condition", "object"):
        command += ["--"+key, job["config"][key]]
    if job["method"] == "ours":
        command += ["--benchmark", "--disk-model", str(args.disk_model)]
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--disk-model", type=Path, default=Path("/home/jyj/Projects/M3T/disk_512.onnx"))
    parser.add_argument("--cache", type=Path, default=Path("/home/jyj/autodl-tmp/M3T-industrial-comparison-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10, help="Timed frames excluded per configuration, accuracy unchanged")
    for key in ("scene", "motion", "condition", "object"):
        parser.add_argument("--"+key, action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)) or not 1 <= args.repeats <= 20:
        parser.error("Methods must be unique; repeats must be 1..20")
    if not 2 <= args.frames <= 500 or not 0 <= args.warmup < args.frames-1:
        parser.error("Need 2..500 input frames and 0 <= warmup < evaluated frames")
    return args


def execute(args, resources):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    hold_lock(resources, output/".run.lock")
    protect_dataset(resources, args.dataset)
    args.cache.mkdir(parents=True, exist_ok=True)
    hold_lock(resources, args.cache/".cache.lock")
    manifest_file = output/"run_manifest.json"
    old = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else None
    if bool(old) != bool(args.resume):
        raise ValueError("Use a new output directory, or --resume with an existing manifest")
    camera = read_camera(args.dataset, allow_motion_subset=True)
    configs = configurations(args.scene, args.motion or camera["motion_modes"], args.condition, args.object)
    if not {c["motion"] for c in configs} <= set(camera["motion_modes"]):
        raise ValueError("Selected motion missing from this dataset")
    if "ours" in args.methods:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Select your allocated GPU via CUDA_VISIBLE_DEVICES")
        if sha256(args.disk_model) != DISK_SHA256:
            raise ValueError("DISK model differs from the tested RBOT model")
    if not shutil.which("xvfb-run"):
        raise ValueError("Source the existing rootless environment to enable xvfb-run")
    identity = {"dataset": str(args.dataset.resolve()), "cache": str(args.cache.resolve()), "methods": list(args.methods), "configurations": configs,
                "frames": args.frames, "repeats": args.repeats, "warmup": args.warmup,
                "builds": {m: verified_build(KIT, m) for m in args.methods},
                "disk_model": str(args.disk_model.resolve()) if "ours" in args.methods else None,
                "disk_model_sha256": DISK_SHA256 if "ours" in args.methods else None,
                "scripts_sha256": {p.relative_to(KIT).as_posix(): sha256(p) for p in KIT.rglob("*.py")
                                   if "sources" not in p.relative_to(KIT).parts and "build" not in p.relative_to(KIT).parts},
                "runtime_policy": {k: os.environ.get(k, "") for k in
                                   ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM",
                                    "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS")}}
    if old and old["identity"] != identity:
        raise ValueError("Resume identity changed (source/binary/scripts/data path/selection/thread policy)")
    manifest = {"schema_version": 1, "status": "validating", "pid": os.getpid(), "identity": identity,
                "started_at": old["started_at"] if old else utc(), "updated_at": utc(), "output": str(output),
                "completed_jobs": list(old.get("completed_jobs", [])) if old else [],
                "expected_jobs": len(configs)*len(args.methods)*args.repeats,
                "expected_evaluated_frames_per_method": len(configs)*(args.frames-1)*args.repeats,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "method_order": "rotate first method for each configuration and repeat"}
    if old and "dataset_fingerprint" in old:
        manifest["dataset_fingerprint"] = old["dataset_fingerprint"]
    write_json(manifest_file, manifest)
    started = time.monotonic(); results = []
    try:
        print("Validating poses, CAD textures and PNG headers; holding rendering read locks...", flush=True)
        validation = preflight(args.dataset, configs, args.frames, allow_motion_subset=True)
        if old and old.get("dataset_fingerprint") != validation["fingerprint"]:
            raise ValueError("Dataset changed; start a new benchmark (do not resume v0.8 into v0.9)")
        manifest["dataset_fingerprint"] = validation["fingerprint"]
        write_json(output/"input_validation.json", validation)
        write_json(manifest_file, manifest)
        for cmd, filename in ((["nvidia-smi"], "gpu.txt"), (["lscpu"], "cpu.txt"),
                              (["uname", "-a"], "system.txt"),
                              (["xvfb-run", "-a", "glxinfo", "-B"], "opengl.txt")):
            with (output/filename).open("w", encoding="utf-8") as stream:
                try:
                    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=False)
                except FileNotFoundError as error:
                    stream.write(str(error))
        scope = "all declared motions, 150 configurations" if len(configs) == 150 and args.frames == 500 else (
                 "v0.9 moving subset, 75 configurations" if len(configs) == 75 and camera["motion_modes"] == ["camera_and_objects_motion"] and args.frames == 500 else "selected subset / smoke test")
        completed = set(manifest["completed_jobs"])
        expected_keys = {j["key"] for j in jobs(configs, args.methods, args.repeats)}
        if not completed <= expected_keys:
            raise ValueError("Unexpected completed job in manifest")
        for number, job in enumerate(jobs(configs, args.methods, args.repeats), 1):
            directory = output/"configurations"/job["key"]
            directory.mkdir(parents=True, exist_ok=True)
            method = job["method"]
            manifest.update(status="running", job=number, current=job, updated_at=utc())
            write_json(manifest_file, manifest)
            if job["key"] not in completed:
                cache = args.cache/validation["fingerprint"][:16]/method/identity["builds"][method]["binary_sha256"][:16]
                cache.mkdir(parents=True, exist_ok=True)
                command = command_for(job, args, cache, directory)
                write_json(directory/"command.json", command)
                print("[%d/%d] %s" % (number, manifest["expected_jobs"], job["key"]), flush=True)
                with (directory/"evaluation.log").open("a", encoding="utf-8") as log:
                    log.write("\n=== attempt %s ===\n" % utc()); log.flush()
                    begin = time.monotonic()
                    try:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as error:
                        log.flush()
                        with (directory/"evaluation.log").open(encoding="utf-8", errors="replace") as errors:
                            from collections import deque
                            tail = "".join(deque(errors, maxlen=35))
                        raise RuntimeError("%s exited with status %s. Full log: %s\n%s" %
                                           (method, error.returncode, directory/"evaluation.log", tail)) from error
                write_json(directory/"process_time.json", {"elapsed_s": time.monotonic()-begin,
                           "scope": "includes startup, setup/cache, inference session loading, tracking and output"})
            item = analyze(directory, args.dataset, job["config"], args.frames)
            timing_rows(item["pose_csv"], args.warmup)
            item.update(method=method, repeat=job["repeat"])
            write_json(directory/"metrics.json", item); results.append(item)
            if job["key"] not in manifest["completed_jobs"]:
                manifest["completed_jobs"].append(job["key"])
            manifest.update(completed_evaluated_frames=sum(r["frames"] for r in results), updated_at=utc())
            write_json(manifest_file, manifest)
            print("  %.3f%%; audit %.3f%%; invalid %d" %
                  (item["success_percent"], item["recomputed_success_percent"], item["invalid_estimate_count"]), flush=True)
            # Save useful reports even if a later job fails.
            if number % (10*len(args.methods)) == 0:
                summarize(output, results, args.warmup, identity, scope, partial=True)
        report = summarize(output, results, args.warmup, identity, scope)
        if any(r["frames"] != manifest["expected_evaluated_frames_per_method"] for r in report["methods"].values()):
            raise ValueError("Unequal final success denominator")
        manifest.update(status="complete", updated_at=utc(), elapsed_this_attempt_s=time.monotonic()-started)
        write_json(manifest_file, manifest)
        print("Complete: "+str(output), flush=True)
    except BaseException as error:
        manifest.update(status="failed", updated_at=utc(), error=str(error))
        write_json(manifest_file, manifest)
        raise


def main(argv=None):
    with contextlib.ExitStack() as resources:
        execute(parse_args(argv), resources)


if __name__ == "__main__":
    main()
