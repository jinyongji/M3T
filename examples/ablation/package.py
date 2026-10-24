"""Ship a separate runner and data readers; reuse frozen server tracker sources."""
import ast
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def package():
    design=json.loads((ROOT/'doc/ablation-plan-20261023.json').read_text(encoding='utf-8'))
    variants=design['core_variants']+[design['additional_priority_control']]
    for v in variants:v['flags']=''.join(str(int(x)) for x in v['modules'].values())
    core={v['file']:v['rbot_frozen_sha256'] for v in design['source']['files'] if v['matches_rbot_frozen'] is True}
    for name,value in [('variants.json',variants),('expected_core.json',core)]:
        (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files={p.name:p.read_bytes().replace(b'\r\n',b'\n') for pattern in ['ablation*.py','ablation_cli.cpp','*.sh','CMakeLists.txt','variants.json','expected_core.json'] for p in HERE.glob(pattern)}
    for parent,names in [('industrial',['dataset.py']),('rbot_compare',['rbot_data.py','rbot_metrics.py'])]:
        for name in names:files['support/'+name]=(HERE.parent/parent/name).read_bytes().replace(b'\r\n',b'\n')
    files['README.md']=(ROOT/'doc/三数据集九组消融服务器说明-20261023.md').read_bytes().replace(b'\r\n',b'\n')
    files['A0-regression-v2.md']=(ROOT/'doc/消融A0回归差异与v2修复说明-20261023.md').read_bytes().replace(b'\r\n',b'\n')
    for name,raw in files.items():
        if name.endswith('.py'):ast.parse(raw.decode('utf-8-sig'),filename=name)
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose='Nine controlled variants on three fixed datasets; parallel accuracy, no formal timing claim',
                  version='v2-video-only-arithmetic-restoration; server strict regression pending',
                  original_design_sha256=hashlib.sha256((ROOT/'doc/ablation-plan-20261023.json').read_bytes()).hexdigest(),
                  sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in sorted(files.items())})
    files['payload_manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    output=ROOT/'m3t-ablation-v2-kit.tar'
    with tarfile.open(output,'w') as archive:
        for name,raw in sorted(files.items()):
            member=tarfile.TarInfo('m3t-ablation-v2/'+name);member.size=len(raw);member.mode=0o755 if name.endswith('.sh') else 0o644
            archive.addfile(member,io.BytesIO(raw))
    with tarfile.open(output) as archive:
        assert len(archive.getmembers())==len(files)
        for m in archive.getmembers():assert archive.extractfile(m).read()==files[m.name.split('/',1)[1]]
    return dict(path=str(output),files=len(files),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':print(json.dumps(package(),ensure_ascii=False,indent=2))
