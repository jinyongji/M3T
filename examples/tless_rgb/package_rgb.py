import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def package():
    files={p.name:p.read_bytes().replace(b'\r\n',b'\n') for pattern in ['rgb_*.py','rgb_*.h','rgb_*.cpp','*.sh','CMakeLists.txt','variants.json'] for p in HERE.glob(pattern)}
    files['README.md']=(ROOT/'doc/T-LESS无纹理RGB改进实验与文献核查-20261024.md').read_bytes().replace(b'\r\n',b'\n')
    for name,raw in files.items():
        if name.endswith('.py'):ast.parse(raw.decode('utf-8-sig'),filename=name)
    manifest=dict(version='rgb-robustness-v1-experimental',base_tag='v2026.10.24-evaluation-baseline',
        sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
    files['payload_manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    output=ROOT/'tless-rgb-experiment-kit.tar'
    with tarfile.open(output,'w') as t:
        for name,raw in sorted(files.items()):
            m=tarfile.TarInfo('tless-rgb-experiment/'+name);m.size=len(raw);m.mode=0o755 if name.endswith('.sh') else 0o644;t.addfile(m,io.BytesIO(raw))
    with tarfile.open(output) as t:
        for m in t.getmembers():assert t.extractfile(m).read()==files[m.name.split('/',1)[1]]
    return dict(path=str(output),files=len(files),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(package(),ensure_ascii=False,indent=2))
