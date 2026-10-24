"""Compare the supplied original M3T with the current working source tree."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def inventory(root):
    result = {}
    for folder in ("include", "src", "examples"):
        for p in (Path(root)/folder).rglob("*"):
            if (p.is_file() and p.suffix in (".cpp", ".h", ".txt")
                    and "industrial_compare" not in p.parts):
                result[p.relative_to(root).as_posix()] = p.read_bytes().replace(b"\r\n", b"\n")
    return result


def compare(original, current):
    a, b = inventory(original), inventory(current)
    changes = []
    for name in sorted(set(a)|set(b)):
        if a.get(name) == b.get(name):
            continue
        diff = list(difflib.unified_diff(a.get(name, b"").decode("utf-8-sig").splitlines(True),
                                        b.get(name, b"").decode("utf-8-sig").splitlines(True)))
        changes.append({"file": name, "added": sum(x.startswith("+") and not x.startswith("+++") for x in diff),
                        "removed": sum(x.startswith("-") and not x.startswith("---") for x in diff),
                        "original_sha256_lf": hashlib.sha256(a[name]).hexdigest() if name in a else None,
                        "current_sha256_lf": hashlib.sha256(b[name]).hexdigest() if name in b else None})
    return {"original": str(original), "current": str(current),
            "scope": "include/src/examples C++ headers/sources and CMakeLists; comparison adapter directory excluded; CRLF normalized to LF",
            "unchanged_files": sum(a.get(k) == b.get(k) for k in set(a)&set(b)), "changed_or_new_files": len(changes),
            "changes": changes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.original, args.current)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Unchanged:", report["unchanged_files"], "changed/new:", report["changed_or_new_files"])
