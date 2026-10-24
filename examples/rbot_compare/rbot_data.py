"""RBOT RGB input contract. Source meshes/poses are in millimetres; never rewrite them."""
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct

BODIES = 'ape bakingsoda benchviseblue broccolisoup cam can cat clown cube driller duck eggbox glue iron koalacandy lamp phone squirrel'.split()
SEQUENCES = ['a_regular', 'b_dynamiclight', 'c_noisy', 'd_occlusion']
METHODS = ['rbgt', 'srt3d', 'icg_rgb', 'm3t_original_rgb', 'ours']
LABELS = dict(zip(METHODS, ['RBGT', 'SRT3D', 'ICG', 'ICG+', 'M3T (ours final)']))
INTRINSICS = dict(fx=650.048, fy=647.183, cx=323.828, cy=256.823, width=640, height=512)
DISK_SHA256 = '5ff1ecfd9a31009a12d0e73ec76b476291756fe101b0417e3d0cc90a0dd42648'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def determinant(r):
    a, b, c, d, e, f, g, h, i = r
    return a * (e*i-f*h) - b * (d*i-f*g) + c * (d*h-e*g)


def read_poses(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            values = [float(x) for x in line.split()]
        except ValueError:
            if index == 0:
                continue  # Original RBOT column header.
            raise ValueError('Invalid RBOT pose line %d' % (index+1))
        if len(values) != 12 or not all(math.isfinite(x) for x in values):
            raise ValueError('Invalid RBOT pose row')
        r = values[:9]
        if abs(determinant(r)-1) > 1e-3 or any(abs(sum(r[k*3+i]*r[k*3+j] for k in range(3))-int(i == j)) > 1e-3
                                             for i in range(3) for j in range(3)):
            raise ValueError('Invalid RBOT rotation')
        result.append(r + [x * .001 for x in values[9:]])
    if len(result) != 1001:
        raise ValueError('RBOT poses_first.txt must contain 1001 poses')
    return result


def configurations(bodies=None, sequences=None):
    return [dict(body=b, object=b, sequence=q, condition=q, scene='RBOT', motion='rbot', id=q+'__'+b)
            for q in (sequences or SEQUENCES) for b in (bodies or BODIES)]


def image_path(root, config, index):
    return Path(root) / config['body'] / 'frames' / ('%s%04d.png' % (config['sequence'], index))


def png_header(path):
    with Path(path).open('rb') as stream:
        h = stream.read(33)
    if (len(h) != 33 or h[:8] != b'\x89PNG\r\n\x1a\n' or h[12:16] != b'IHDR'
            or struct.unpack('>II', h[16:24]) != (640, 512) or h[24:26] != bytes([8, 2])):
        raise ValueError('Expected 640x512 uint8 RGB RBOT PNG: ' + str(path))
    return h


def model_assets(root, body):
    """Only the selected OBJ, its material libraries and their diffuse images.

    RBOT folders may also contain gigabytes of derived depth/templates/output;
    these are not algorithm inputs and must not affect fingerprints or resume.
    """
    root = Path(root)
    directory = root/body
    mesh = directory/(body+'.obj')
    selected = {mesh}
    materials = []
    for line in mesh.read_text(encoding='utf-8-sig').splitlines():
        if line.lstrip().startswith('mtllib'):
            fields = shlex.split(line, comments=True)
            if fields and fields[0] == 'mtllib':
                materials.extend(directory/name for name in fields[1:])
    for material in materials:
        selected.add(material)
        for line in material.read_text(encoding='utf-8-sig').splitlines():
            if line.lstrip().startswith('map_Kd'):
                fields = shlex.split(line, comments=True)
                if not fields or fields[0] != 'map_Kd':
                    continue
                if len(fields) < 2:
                    raise ValueError('Missing diffuse texture filename: '+str(material))
                # TinyObjLoader's diffuse texture is relative to the OBJ folder.
                # Standard texture options precede the final filename.
                name = fields[-1] if fields[1].startswith('-') else ' '.join(fields[1:])
                selected.add(directory/name)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(selected)}


def preflight(root, configs, frames):
    root = Path(root)
    poses = read_poses(root/'poses_first.txt')
    h = hashlib.sha256()
    assets = {}
    for body in sorted({r['body'] for r in configs}):
        directory = root/body
        if not (directory/(body+'.obj')).is_file():
            raise ValueError('Missing RBOT mesh: ' + body)
        assets.update(model_assets(root, body))
    assets_hash = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    model_hashes = {body: hashlib.sha256(json.dumps(dict(root=str(root), assets={k: v for k, v in assets.items() if k.startswith(body+'/')}),
                                                   sort_keys=True).encode()).hexdigest()
                    for body in sorted({r['body'] for r in configs})}
    h.update(json.dumps(dict(assets=assets, pose_sha256=sha256(root/'poses_first.txt'),
                            root=str(root), intrinsics=INTRINSICS), sort_keys=True).encode())
    for config in configs:
        for index in range(frames):
            p = image_path(root, config, index)
            header = png_header(p)
            stat = p.stat()
            h.update(json.dumps([p.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns]).encode())
            h.update(header)
    return {'input_fingerprint': h.hexdigest(), 'model_fingerprint': assets_hash, 'model_fingerprints': model_hashes,
            'pose_sha256': sha256(root/'poses_first.txt'), 'asset_sha256': assets,
            'fingerprint_scope': 'models/pose SHA256; all selected PNG IHDR, size and mtime; not full PNG pixel hashes'}, poses


def write_normalized_poses(path, poses):
    text = ''.join(' '.join(format(x, '.17g') for x in p) + '\n' for p in poses)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Existing normalized input changed: ' + str(path))
    if not path.exists():
        path.write_text(text, encoding='utf-8')


def old_run_assessment(path):
    path = Path(path)
    files = sorted((path/'results/poses').glob('poses_*.csv'))
    import csv
    timing = True
    for p in files:
        with p.open(encoding='utf-8-sig', newline='') as stream:
            fields = csv.DictReader(stream).fieldnames or []
        timing = timing and {'decode_us', 'compute_wall_us', 'tracking_wall_us'} <= set(fields)
    result = {'path': str(path), 'pose_files': len(files), 'has_comparable_timing_columns': bool(files) and timing,
              'use': 'historical accuracy reference only; do not combine old native-cycle timing with compute-wall timing',
              'rerun_ours_for_paired_timing': not files or not timing}
    manifest, summary = path/'run_manifest.json', path/'summary.json'
    if manifest.is_file() and summary.is_file():
        m = json.loads(manifest.read_text(encoding='utf-8-sig'))
        s = json.loads(summary.read_text(encoding='utf-8-sig'))
        if (m.get('Status') == 'complete' and m.get('Variant') == 'final' and m.get('ExpectedConfigurations') == 72
                and s.get('configurations') == 72 and s.get('frames') == 72000 and len(files) == 72):
            rate = s.get('overall_success_percent')
            if isinstance(rate, (float, int)) and math.isfinite(rate) and 0 <= rate <= 100:
                result.update(historical_native_success_percent=rate, run_manifest_sha256=sha256(manifest),
                              original_binary_sha256=m.get('ExecutableSHA256'), disk_model_sha256=m.get('DiskModelSHA256'))
    return result
