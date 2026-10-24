"""Bind a binary to the precise copied source and build configuration."""
import json
import subprocess
import sys
from pathlib import Path

SHARED = Path(__file__).parent/"shared"
sys.path.insert(0, str(SHARED if SHARED.is_dir() else Path(__file__).parent.parent/"industrial"))
from dataset import sha256, write_json

SOURCE_DIRS = {"rbgt": "RBGT", "srt3d": "SRT3D", "icg_rgb": "ICG", "m3t_original_rgb": "M3T"}


def source_files(kit, method):
    kit = Path(kit)
    root = kit/"sources/current_m3t" if method == "ours" else kit/"sources/upstream"/SOURCE_DIRS[method]
    if not (root/"src/CMakeLists.txt").is_file():
        raise ValueError("Missing packaged source: "+str(root))
    result = {"source/"+p.relative_to(root).as_posix(): sha256(p)
              for folder in ("src", "include", "third_party", "examples")
              for p in (root/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    for filename in ("CMakeLists.txt", "baseline.cpp"):
        result[filename] = sha256(kit/filename)
    return result


def record(kit, method):
    kit = Path(kit)
    build = kit/"build"/method
    binary = build/"evaluate_industrial_baseline"
    compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    write_json(build/"build_identity.json", {"method": method, "binary_sha256": sha256(binary),
               "source_sha256": source_files(kit, method), "cmake_cache_sha256": sha256(build/"CMakeCache.txt"),
               "compiler": compiler, "mode": "Release -O3; C++17; same wrapper and host compiler"})


def prepare(kit, method):
    """Make changed inputs newer than old objects, even after tar/clock skew."""
    kit = Path(kit)
    old_file = kit/"build"/method/"build_identity.json"
    if not old_file.is_file():
        return []
    old = json.loads(old_file.read_text(encoding="utf-8")).get("source_sha256", {})
    current = source_files(kit, method)
    root = kit/"sources/current_m3t" if method == "ours" else kit/"sources/upstream"/SOURCE_DIRS[method]
    changed = [name for name in current if old.get(name) != current[name]]
    for name in changed:
        path = root/name[len("source/"):] if name.startswith("source/") else kit/name
        path.touch()
    return changed


def verified_build(kit, method):
    kit = Path(kit)
    build = kit/"build"/method
    value = json.loads((build/"build_identity.json").read_text(encoding="utf-8"))
    if (value["method"] != method or value["binary_sha256"] != sha256(build/"evaluate_industrial_baseline")
            or value["source_sha256"] != source_files(kit, method)
            or value["cmake_cache_sha256"] != sha256(build/"CMakeCache.txt")):
        raise ValueError("Build/source changed; rebuild before evaluating: "+method)
    return value


if __name__ == "__main__":
    if sys.argv[1] == "--prepare":
        changed = prepare(Path(__file__).parent, sys.argv[2])
        print("Changed build inputs refreshed:", len(changed))
    else:
        record(Path(__file__).parent, sys.argv[1])
