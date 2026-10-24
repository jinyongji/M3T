#!/usr/bin/env python3
"""Build a self-contained source kit; do not edit either source repository."""
import argparse
import datetime
import difflib
import hashlib
import json
import tarfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_package(upstream, current, destination):
    upstream, current, destination = Path(upstream), Path(current), Path(destination)
    files = {}
    changes = []
    source_manifest = {}

    def add_source(root, prefix, patched=False):
        selected = []
        for folder in ("src", "include", "third_party"):
            selected += [p for p in (root/folder).rglob("*") if p.is_file()]
        selected += [p for p in root.iterdir() if p.is_file() and
                     (p.name == "CMakeLists.txt" or p.name.lower().startswith(("readme", "license")))]
        # Preserve the original RGB example configurations as parameter provenance.
        selected += [p for p in (root/"examples").glob("*") if p.suffix in (".cpp", ".h")]
        for path in sorted(selected):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            before = after = path.read_bytes()
            if patched and relative == "src/CMakeLists.txt":
                old = b'"${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"'
                assert old in before, "Unexpected ORT link clause"
                after = before.replace(old, b'"${ONNXRUNTIME_LIBRARY}"')
            if patched and relative == "src/disk_feature_extractor.cpp":
                before_lf = before.replace(b"\r\n", b"\n")
                old = b"      Ort::ThrowOnError(\n          OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));"
                new = b"      OrtCUDAProviderOptions cuda_options{};\n      cuda_options.device_id = 0;\n      options.AppendExecutionProvider_CUDA(cuda_options);"
                assert old in before_lf, "Unexpected CUDA provider clause"
                after = before_lf.replace(old, new)
            name = prefix+"/"+relative
            files[name] = after
            source_manifest[name] = {"original_sha256": digest(before), "packaged_sha256": digest(after),
                                     "portability_patch": before != after}
            if before != after:
                changes.extend(difflib.unified_diff(before.decode("utf-8-sig").splitlines(True),
                    after.decode("utf-8-sig").splitlines(True), fromfile="current/"+relative, tofile="linux/"+relative))

    for method in ("RBGT", "SRT3D", "ICG", "M3T"):
        add_source(upstream/method, "sources/upstream/"+method)
    add_source(current, "sources/current_m3t", patched=True)
    files["sources/upstream/LICENSE"] = (upstream/"LICENSE").read_bytes()
    files["sources/upstream/ICG+-readme.md"] = (upstream/"ICG+/readme.md").read_bytes()
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in (".py", ".sh", ".cpp", ".txt") and p.name != "package.py":
            files[p.name] = p.read_bytes().replace(b"\r\n", b"\n")
    for name in ("dataset.py", "metrics.py"):
        files["shared/"+name] = (current/"examples/industrial"/name).read_bytes().replace(b"\r\n", b"\n")
    for name in ("原始跟踪器与改进M3T代码比较及速度优化.md", "工业数据集多算法服务器评测说明.md"):
        files["docs/"+name] = (current/"doc"/name).read_bytes()
    files["docs/supplied_m3t_source_comparison.json"] = (current/"doc/supplied_m3t_source_comparison.json").read_bytes()
    files["linux_portability.patch"] = "".join(changes).encode("utf-8")
    files["source_manifest.json"] = (json.dumps({"schema_version": 1,
        "upstream_location_at_packaging": str(upstream), "current_location_at_packaging": str(current),
        "upstream_policy": "byte-identical files from supplied tree; no algorithm changes",
        "current_policy": "working tree snapshot; two Linux ORT portability edits only",
        "files": source_manifest}, indent=2, ensure_ascii=False)+"\n").encode("utf-8")
    files["payload_manifest.json"] = (json.dumps({"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256": {k: digest(v) for k, v in sorted(files.items())}}, indent=2, ensure_ascii=False)+"\n").encode("utf-8")
    # Tar avoids the server's system Python ZIP/zlib limitation. GNU tar handles gzip.
    import io
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo("industrial-tracker-comparison/"+name)
            item.size = len(data); item.mode = 0o755 if name.endswith(".sh") else 0o644
            item.mtime = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
            archive.addfile(item, io.BytesIO(data))
    return {"path": str(destination), "bytes": destination.stat().st_size, "files": len(files),
            "sha256": digest(destination.read_bytes()), "patched_files": [k for k, v in source_manifest.items() if v["portability_patch"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("F:/projects/3DObjectTracking-master"))
    parser.add_argument("--current", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("industrial-tracker-comparison.tar.gz"))
    args = parser.parse_args()
    print(json.dumps(make_package(args.upstream, args.current, args.output), indent=2, ensure_ascii=False))
