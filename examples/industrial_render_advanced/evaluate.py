"""Evaluate advanced-v1 with the existing five verified tracker binaries.

Read-only symlink views adapt path layout, never pixels, geometry or poses.
The native neutral condition alias is recorded separately from true conditions.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

METHODS = ["rbgt", "srt3d", "icg_rgb", "m3t_original_rgb", "ours"]
NAME = "工业跟踪进阶数据集-v1.0-20261018"
NATIVE_CONDITION = "01_normal"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def link(source, destination, replace=False):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            if not replace: raise ValueError("View link changed: "+str(destination))
            destination.unlink()
            destination.symlink_to(source, target_is_directory=source.is_dir())
    elif destination.exists():
        raise ValueError("Unexpected nonsymlink in view: "+str(destination))
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def native_config(row):
    return {"scene": row["scene"], "motion": row["motion_mode"], "condition": NATIVE_CONDITION,
            "object": row["object"], "id": row["sequence_id"]+"__"+row["object"]}


def create_view(root, row, view):
    root, view = Path(root).resolve(), Path(view)
    view.mkdir(parents=True, exist_ok=True)
    config = native_config(row)
    # camera.json remains the real nine-object metadata. The native C++
    # loaders only use its calibrated K, dimensions, units and pose convention.
    link(root/"cameras", view/"cameras"); link(root/"objects", view/"objects")
    relative = Path(row["scene"])/row["motion_mode"]/NATIVE_CONDITION
    link(root/row["rgb_path"], view/"images"/relative, replace=True)
    link(root/row["poses_path"], view/"poses"/relative, replace=True)
    return config


def select_rows(index, focus_only=False, smoke=False):
    rows = index["configurations"]
    if focus_only: rows = [r for r in rows if r["is_focus"]]
    if smoke:
        # One reference, an occlusion plateau job and a motion intervention.
        rows = [r for r in rows if r["scene"] == "scene_04" and r["trajectory"] == "t01"
                and r["condition"] in ("B00", "O03", "M02") and r["is_focus"]]
    if not rows: raise ValueError("No selected advanced-v1 configurations")
    return rows


def input_identity(data, rows, frames, png_header):
    files = {"cameras/camera.json": digest_file(data/"cameras/camera.json"),
             "cameras/sequence_index.csv": digest_file(data/"cameras/sequence_index.csv")}
    for p in (data/"objects").rglob("*"):
        if p.is_file() and p.suffix.lower() != ".blend": files[p.relative_to(data).as_posix()] = digest_file(p)
    for row in rows:
        pose = data/row["ground_truth_file"]
        files[row["ground_truth_file"]] = digest_file(pose)
        for i in range(1, frames+1):
            p = data/row["rgb_path"]/("%06d.png" % i)
            relative = p.relative_to(data).as_posix()
            if relative in files: continue
            header = png_header(p); stat = p.stat()
            files[relative] = [stat.st_size, stat.st_mtime_ns, hashlib.sha256(header).hexdigest()]
    h = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return h


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Extracted dataset directory above 交付物")
    p.add_argument("--kit", type=Path, default=Path("/home/jyj/Projects/industrial-tracker-comparison"))
    p.add_argument("--disk-model", type=Path, default=Path("/home/jyj/Projects/M3T/disk_512.onnx"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=500); p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--focus-only", action="store_true"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.resume:
        saved = read_json(args.output/"run_manifest.json")["identity"]["launch"]
        for key in ("frames", "warmup", "focus_only", "smoke"): setattr(args, key, saved[key])
        for key in ("root", "kit", "disk_model"): setattr(args, key, Path(saved[key]))
    if not 2 <= args.frames <= 500 or not 0 <= args.warmup < args.frames-1: p.error("Invalid frame count/warmup")
    sys.path.insert(0, str(args.kit)); sys.path.insert(0, str(args.kit/"shared"))
    from dataset import sha256, write_json, png_header, DISK_SHA256
    from metrics import analyze
    from report import summarize, timing_rows, LABELS
    LABELS.update(icg_rgb="ICG", m3t_original_rgb="ICG+")
    from provenance import verified_build
    from compare import hold_lock
    root = args.root.resolve(); data = root/"交付物"/NAME; output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"run_manifest.json"
    with contextlib.ExitStack() as resources:
        hold_lock(resources, output/".run.lock")
        state = root/"中间结果文件/run_state_advanced_v1"
        hold_lock(resources, state/"render.lock", shared=True, existing_only=True)
        audit = read_json(state/"validation/final_audit.json")
        if audit.get("status") != "passed" or audit.get("scope") != "complete_dataset" or audit.get("images") != 54000:
            raise ValueError("Complete rendering audit must pass before evaluation")
        if read_json(state/"independent_audit/inputs.json").get("status") != "passed":
            raise ValueError("Run render.sh check before evaluating")
        scene_audit = read_json(state/"independent_audit/scenes.json")
        if (scene_audit.get("status"), scene_audit.get("scope"), scene_audit.get("sequences_checked"),
                scene_audit.get("total_frame_checks")) != ("passed", "all integer frames", 108, 54000):
            raise ValueError("Complete the independent Blender scene audit before evaluating")
        index = read_json(state/"independent_audit/evaluation_index.json")
        rows = select_rows(index, args.focus_only, args.smoke)
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or sha256(args.disk_model) != DISK_SHA256:
            raise ValueError("Select allocated GPU and use the verified DISK model")
        if not shutil.which("xvfb-run"): raise ValueError("Source the existing rootless dependency environment")
        # Same cache lock as the original comparison and rendering wrapper.
        shared_cache = Path(os.environ.get("M3T_COMPARE_CACHE", "/home/jyj/autodl-tmp/M3T-industrial-comparison-cache"))
        shared_cache.mkdir(parents=True, exist_ok=True); hold_lock(resources, shared_cache/".cache.lock")
        configs = [dict(native_config(r), condition=r["condition"], trajectory=r["trajectory"],
                        pair_id=r["pair_id"], is_focus=r["is_focus"], split=r["split"]) for r in rows]
        identity = {"dataset": str(data), "methods": METHODS, "method_labels": {m: LABELS[m] for m in METHODS},
                    "configurations": configs, "repeats": 1,
                    "frames": args.frames, "warmup": args.warmup, "builds": {m: verified_build(args.kit, m) for m in METHODS},
                    "input_fingerprint": input_identity(data, rows, args.frames, png_header),
                    "adapter_sha256": digest_file(__file__), "index_sha256": digest_file(state/"independent_audit/evaluation_index.json"),
                    "scripts_sha256": {x.relative_to(args.kit).as_posix(): digest_file(x)
                                       for base in (args.kit, args.kit/"shared") for x in base.glob("*.py")},
                    "launch": {"root": str(root), "kit": str(args.kit.resolve()), "disk_model": str(args.disk_model.resolve()),
                               "frames": args.frames, "warmup": args.warmup, "focus_only": args.focus_only, "smoke": args.smoke},
                    "threads": {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENCV_FOR_THREADS_NUM", "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS")},
                    "native_alias": "01_normal is only a native path/CLI alias; condition never selects an algorithm branch"}
        old = read_json(manifest_path) if manifest_path.exists() else None
        if bool(old) != args.resume: raise ValueError("Choose new output or resume an existing run")
        if old and old["identity"] != identity: raise ValueError("Source/binary/data/selection/adapter changed; start a new run")
        manifest = {"status": "running", "pid": os.getpid(), "output": str(output), "identity": identity,
                    "completed_jobs": list(old["completed_jobs"]) if old else [], "expected_jobs": len(rows)*5,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "adapter_mapping": "original scene/trajectory/condition kept in mapping.json; native read-only symlink view uses 01_normal"}
        write_json(manifest_path, manifest); results = []; completed = set(manifest["completed_jobs"])
        cache = output.parent/"model_cache"/identity["input_fingerprint"][:16]
        try:
            for number, (row, config) in enumerate(zip(rows, configs)):
                # Keep CAD paths stable: legacy model caches compare geometry
                # path strings. Only image/pose symlinks change under the run lock.
                view = output/"input_view"
                native = create_view(data, row, view)
                order = METHODS[number % 5:]+METHODS[:number % 5]
                for method in order:
                    key = config["id"]+"/"+method; directory = output/"configurations"/key
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory/"mapping.json", {"original": row, "native": native, "view": str(view),
                              "note": "Read-only path aliases; original camera metadata and files preserved"})
                    manifest.update(current=dict(config, method=method), job=len(results)+1)
                    write_json(manifest_path, manifest)
                    if key not in completed:
                        model_cache = cache/method/identity["builds"][method]["binary_sha256"][:16]
                        model_cache.mkdir(parents=True, exist_ok=True)
                        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24",
                               str(args.kit/"build"/method/"evaluate_industrial_baseline"),
                               "--dataset", str(view), "--cache", str(model_cache), "--output", str(directory),
                               "--frames", str(args.frames)]
                        for k in ("scene", "motion", "condition", "object"): cmd += ["--"+k, native[k]]
                        if method == "ours": cmd += ["--benchmark", "--disk-model", str(args.disk_model)]
                        write_json(directory/"command.json", cmd)
                        print("[%d/%d] %s" % (len(results)+1, len(rows)*5, key), flush=True)
                        with (directory/"evaluation.log").open("a", encoding="utf-8") as log:
                            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                    item = analyze(directory, view, native, args.frames)
                    timing_rows(item["pose_csv"], args.warmup)
                    item.update(config, method=method, repeat=1); results.append(item)
                    write_json(directory/"metrics.json", item)
                    if key not in manifest["completed_jobs"]: manifest["completed_jobs"].append(key)
                    write_json(manifest_path, manifest)
                    print("  success %.3f%%; audit %.3f%%" % (item["success_percent"], item["recomputed_success_percent"]), flush=True)
            summarize(output, results, args.warmup, identity, "advanced-v1: "+str(len(rows))+" target configurations")
            focus_output = output/"focus"; focus_output.mkdir(exist_ok=True)
            focus_identity = dict(identity, configurations=[c for c in configs if c["is_focus"]])
            summarize(focus_output, [r for r in results if r["is_focus"]], args.warmup, focus_identity,
                      "advanced-v1 focus objects only")
            manifest.update(status="complete"); write_json(manifest_path, manifest)
            print("Complete: "+str(output), flush=True)
        except BaseException as error:
            manifest.update(status="failed", error=str(error)); write_json(manifest_path, manifest); raise


if __name__ == "__main__": main()
