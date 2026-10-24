"""Check packaged file contents before compiling; requires only Python stdlib."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/"payload_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = (root/relative).resolve()
        path.relative_to(root)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Missing/modified packaged input: "+relative)
    return len(manifest["sha256"])


if __name__ == "__main__":
    print("Verified packaged files:", verify(Path(__file__).parent))
