"""Small standalone server kit; reuse existing frozen engines, data and weights."""
import ast
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def package(output):
    files={}
    names=['tracking_data.py','tracking_build.py','tracking_report.py','tracking_run.py','tracking_python.py',
           'tracking_driver.cpp','CMakeLists.txt','run_tracking.sh','build_tracking.sh','expected_sources.json']
    for name in names:files[name]=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
    files['README.md']=(ROOT/'doc/T-LESS五方法服务器评测说明-20261023.md').read_bytes().replace(b'\r\n',b'\n')
    for name,raw in files.items():
        if name.endswith('.py'):ast.parse(raw.decode('utf-8-sig'),filename=name)
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose='Five-method RGB T-LESS static view-ring tracking; GT-init/reset, symmetry scoring, compute-only timing',
                  sha256={n:hashlib.sha256(b).hexdigest() for n,b in sorted(files.items())})
    files['payload_manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    output=Path(output)
    with tarfile.open(output,'w') as archive:
        for name,raw in sorted(files.items()):
            member=tarfile.TarInfo('tless-tracker-comparison/'+name);member.size=len(raw);member.mode=0o755 if name.endswith('.sh') else 0o644
            archive.addfile(member,io.BytesIO(raw))
    with tarfile.open(output,'r') as archive:
        assert len(archive.getmembers())==len(files)
        for member in archive.getmembers():assert archive.extractfile(member).read()==files[member.name.split('/',1)[1]]
    return dict(path=str(output),files=len(files),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':print(json.dumps(package(ROOT/'tless-five-methods-kit.tar'),ensure_ascii=False,indent=2))
