"""Package the small RBOT kit plus T-LESS preparation helpers; no algorithm/data copies."""
import argparse
import ast
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def package(destination):
    files = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and (path.suffix in ['.py', '.sh'] or path.name == 'CMakeLists.txt') and path.name != 'package.py':
            files[path.name] = path.read_bytes().replace(b'\r\n', b'\n')
    files['reformat_summary.py'] = (PROJECT/'examples/industrial_compare/reformat_summary.py').read_bytes().replace(b'\r\n', b'\n')
    for name in ['prepare_tless.py', 'prepare.sh']:
        files['tless/'+name] = (PROJECT/'examples/tless'/name).read_bytes().replace(b'\r\n', b'\n')
    for name in ['RBOT五方法评测与计算计时说明-20261022.md', 'T-LESS五方法真实工业评估方案-20261022.md']:
        files['docs/'+name] = (PROJECT/'doc'/name).read_bytes()
    for name, raw in files.items():
        if name.endswith('.py'):
            ast.parse(raw.decode('utf-8-sig'), filename=name)
    manifest = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'purpose': 'RBOT five-method compute-only benchmark; T-LESS download/input audit only',
                'sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())}}
    files['payload_manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, 'w') as archive:
        for name, raw in sorted(files.items()):
            item = tarfile.TarInfo('rbot-tracker-comparison/'+name)
            item.size = len(raw); item.mode = 0o755 if name.endswith('.sh') else 0o644
            archive.addfile(item, io.BytesIO(raw))
    with tarfile.open(destination, 'r') as archive:
        assert len(archive.getmembers()) == len(files)
        for item in archive.getmembers():
            name = item.name.removeprefix('rbot-tracker-comparison/') if hasattr(str, 'removeprefix') else item.name[len('rbot-tracker-comparison/'):]
            assert archive.extractfile(item).read() == files[name]
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'files': len(files),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'rbot-five-methods-kit.tar')
    args = parser.parse_args()
    print(json.dumps(package(args.output), ensure_ascii=False, indent=2))
