"""Verify packaged helper bytes before a server build."""
import json
from pathlib import Path
from rbot_data import sha256

root = Path(__file__).resolve().parent
manifest = root/'payload_manifest.json'
if manifest.is_file():
    files = json.loads(manifest.read_text(encoding='utf-8'))['sha256']
    for name, value in files.items():
        target = (root/name).resolve()
        target.relative_to(root)
        if not target.is_file() or sha256(target) != value:
            raise ValueError('Packaged helper differs from manifest: '+name)
    print('Helper payload verified: %d files' % len(files))
else:
    print('Source checkout: package manifest not present')
