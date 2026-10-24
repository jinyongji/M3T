"""Install the reviewed adapter payload; preflight every file before writing."""
import datetime
import hashlib
import json
import sys
from pathlib import Path


def normalized(data):
    return data.replace(b"\r\n", b"\n")


def digest(data):
    return hashlib.sha256(normalized(data)).hexdigest()


def install(kit, source, require_linux=True):
    kit, source = Path(kit).resolve(), Path(source).resolve()
    manifest = json.loads((kit / "payload_manifest.json").read_text(encoding="utf-8"))
    if require_linux:
        if "find_library(ONNXRUNTIME_LIBRARY" not in (source / "CMakeLists.txt").read_text(encoding="utf-8-sig"):
            raise ValueError("Expected the already working RBOT Linux project; root CMake lacks its Linux ORT patch")
    changes = []
    for relative, spec in manifest["files"].items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe payload path")
        target = source / rel
        target.resolve().relative_to(source)
        content = normalized((kit / "payload" / rel).read_bytes())
        if digest(content) != spec["new_sha256"]:
            raise ValueError("Corrupt payload: " + relative)
        old = target.read_bytes() if target.exists() else None
        if old is not None and digest(old) == spec["new_sha256"]:
            continue
        if old is None:
            if not spec.get("allow_missing", False):
                raise ValueError("Required base file missing: " + relative)
        elif digest(old) not in spec["allowed_base_sha256"]:
            raise ValueError("Local changes would be overwritten; no files changed: " + relative)
        changes.append((target, rel, old, content))
    if not changes:
        print("Industrial adapter already installed; no files changed")
        return []
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = source / "industrial-eval-backups" / stamp
    # Back up every original byte before the first source mutation.
    for target, rel, old, content in changes:
        if old is not None:
            saved = backup / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(old)
    for target, rel, old, content in changes:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".industrial-install.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    print("Installed %d adapter files. Original files: %s" % (len(changes), backup))
    return changes


if __name__ == "__main__":
    try:
        install(Path(__file__).parent, Path(sys.argv[1]))
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise SystemExit("Install failed: " + str(error))
