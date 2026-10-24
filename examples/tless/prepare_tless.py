#!/usr/bin/env python3
"""Download pinned official T-LESS archives and audit inputs before tracker adaptation."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import subprocess
from urllib.parse import urlsplit
import zipfile

REVISION = '5fd309a04476a842d93abfb584fba9ee7caecdf1'
FILES = {
    'tless_base.zip': (49597, 'dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e'),
    'tless_models.zip': (33468762, '6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928'),
    'tless_test_primesense_all.zip': (8330530161, 'ee2cbf634158cf940383247b45ab663cd6bd9dbe79e45f90eabd79aed5a04f9c'),
}
DEFAULT = '/home/jyj/autodl-tmp/bop_datasets/tless'
SOURCE = 'https://huggingface.co/datasets/bop-benchmark/tless'


def download_endpoint(endpoint=None):
    endpoint = (endpoint if endpoint is not None else os.environ.get('HF_ENDPOINT', 'https://huggingface.co')).strip().rstrip('/')
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ['https', 'http'] or not parsed.netloc or parsed.path
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('HF_ENDPOINT/--endpoint must be an HTTP(S) origin, for example https://hf-mirror.com')
    return endpoint


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def extract_archive(archive, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as source:
        records = []
        for info in source.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsupported archive member: '+info.filename)
            parts = p.parts[1:] if p.parts and p.parts[0] == 'tless' else p.parts
            if not parts:
                continue
            target = root.joinpath(*parts).resolve()
            target.relative_to(root)
            if target == root:
                raise ValueError('Invalid archive target')
            records.append((info, target))
        needed = sum(info.file_size for info, _ in records)
        if shutil.disk_usage(root).free < needed + 2*1024**3:
            raise ValueError('Insufficient space for archive extraction: '+str(archive))
        for info, target in records:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Archive SHA256 is checked before extraction; ZipExtFile also checks CRC.
            temporary = target.with_name(target.name+'.tless-extract-partial')
            with source.open(info) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            temporary.replace(target)
    return len(records)


def download(root, endpoint=None):
    endpoint = download_endpoint(endpoint)
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    archives = root.parent/'tless_archives'; archives.mkdir(parents=True, exist_ok=True)
    environment = {k: v for k, v in os.environ.items() if k not in ['LD_LIBRARY_PATH', 'LD_PRELOAD']}
    manifest = dict(revision=REVISION, source=SOURCE, download_endpoint=endpoint, files={})
    print('Download endpoint: '+endpoint, flush=True)
    for name, (size, digest) in FILES.items():
        destination = archives/name
        if destination.exists() and (destination.stat().st_size != size or sha256(destination) != digest):
            raise ValueError('Existing archive differs from pinned checksum; preserve and inspect it: '+str(destination))
        if not destination.exists():
            if shutil.disk_usage(archives).free < size+2*1024**3:
                raise ValueError('Insufficient free space for download')
            partial = archives/(name+'.partial')
            if partial.exists() and partial.stat().st_size > size:
                raise ValueError('Partial file larger than expected: '+str(partial))
            print('Downloading '+name+' (%d bytes)' % size, flush=True)
            if not partial.exists() or partial.stat().st_size < size:
                subprocess.run(['/usr/bin/curl', '--fail', '--location', '--retry', '8', '--retry-delay', '3',
                                '--connect-timeout', '20', '--continue-at', '-', '--output', str(partial),
                                endpoint+'/datasets/bop-benchmark/tless/resolve/'+REVISION+'/'+name],
                               check=True, env=environment)
            if partial.stat().st_size != size or sha256(partial) != digest:
                raise ValueError('Downloaded archive failed size/SHA256: '+str(partial))
            partial.replace(destination)
        marker = root/('.extracted-'+name+'.json')
        if marker.is_file() and json.loads(marker.read_text()).get('archive_sha256') == digest:
            print('Already extracted '+name, flush=True)
        else:
            print('Extracting '+name, flush=True)
            count = extract_archive(destination, root)
            save(marker, dict(archive_sha256=digest, entries=count))
        manifest['files'][name] = dict(size=size, sha256=digest)
    save(root/'download_manifest.json', manifest)
    print('Download and extraction complete: '+str(root), flush=True)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def valid_rotation(r):
    if len(r) != 9 or not all(math.isfinite(v) for v in r):
        return False
    a,b,c,d,e,f,g,h,i = r
    det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    return abs(det-1) < 1e-3 and all(abs(sum(r[k*3+i]*r[k*3+j] for k in range(3))-int(i == j)) < 1e-3
                                     for i in range(3) for j in range(3))


def audit(root, output):
    root = Path(root)
    model_info = load(root/'models_cad/models_info.json')
    if set(model_info) != {str(i) for i in range(1, 31)}:
        raise ValueError('Expected all 30 CAD models')
    models = []
    for key in sorted(model_info, key=int):
        path = root/'models_cad'/('obj_%06d.ply' % int(key))
        with path.open('rb') as stream:
            header = []
            for _ in range(1000):
                line = stream.readline().decode('ascii').strip(); header.append(line)
                if line == 'end_header':
                    break
            else:
                raise ValueError('Invalid PLY header')
        info = model_info[key]
        if not math.isfinite(info['diameter']) or info['diameter'] <= 0:
            raise ValueError('Invalid diameter')
        models.append(dict(obj_id=int(key), diameter_mm=info['diameter'],
                           discrete_symmetries=len(info.get('symmetries_discrete', [])),
                           continuous_symmetry_axes=len(info.get('symmetries_continuous', [])),
                           ply_properties=[x for x in header if x.startswith('property')], sha256=sha256(path)))
    split = root/'test_primesense'
    scenes = sorted(p for p in split.iterdir() if p.is_dir() and p.name.isdigit())
    if {int(p.name) for p in scenes} != set(range(1, 21)):
        raise ValueError('Need all 20 test scenes; download test_primesense_all rather than only bop19')
    reports = []; total = target_instances = 0
    for scene in scenes:
        cameras = load(scene/'scene_camera.json'); truth = load(scene/'scene_gt.json'); info = load(scene/'scene_gt_info.json')
        images = {int(p.stem): p for p in (scene/'rgb').glob('*.png')}
        if (len(images) != 504 or set(images) != {int(k) for k in cameras}
                or set(cameras) != set(truth) or set(truth) != set(info)):
            raise ValueError('Incomplete full-scene RGB/annotations: '+str(scene))
        ks = set(); extrinsics = multi = 0; shapes = set(); instances = 0
        for image_id, image in sorted(images.items()):
            camera, poses, gtinfo = cameras[str(image_id)], truth[str(image_id)], info[str(image_id)]
            k = camera['cam_K']
            if len(k) != 9 or not all(math.isfinite(v) for v in k) or k[0] <= 0 or k[4] <= 0 or any(abs(a-b) > 1e-7 for a,b in zip(k[6:], [0,0,1])):
                raise ValueError('Invalid per-image K')
            ks.add(tuple(k))
            if 'cam_R_w2c' in camera and 'cam_t_w2c' in camera:
                if not valid_rotation(camera['cam_R_w2c']) or len(camera['cam_t_w2c']) != 3 or not all(math.isfinite(v) for v in camera['cam_t_w2c']):
                    raise ValueError('Invalid camera extrinsics')
                extrinsics += 1
            with image.open('rb') as stream:
                h = stream.read(33)
            if len(h) != 33 or h[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Invalid RGB PNG')
            shape = struct.unpack('>II', h[16:24]); shapes.add(shape)
            if shape != (720, 540):
                raise ValueError('Expected native 720x540 test Primesense RGB; do not rescale silently')
            if len(poses) != len(gtinfo):
                raise ValueError('Pose/info instance index mismatch')
            for pose, detail in zip(poses, gtinfo):
                if (str(pose['obj_id']) not in model_info or not valid_rotation(pose['cam_R_m2c'])
                        or len(pose['cam_t_m2c']) != 3 or not all(math.isfinite(v) for v in pose['cam_t_m2c'])
                        or not 0 <= detail['visib_fract'] <= 1):
                    raise ValueError('Invalid instance pose/visibility')
            if any(count > 1 for count in Counter(p['obj_id'] for p in poses).values()):
                multi += 1
            instances += len(poses)
        record = dict(scene_id=int(scene.name), images=len(images), gt_instances=instances,
                      unique_intrinsics=len(ks), images_with_camera_extrinsics=extrinsics,
                      images_with_repeated_obj_id=multi, rgb_sizes=[list(x) for x in sorted(shapes)])
        reports.append(record); total += len(images); target_instances += instances
        print(json.dumps(record), flush=True)
    result = dict(status='passed_input_structure', dataset='T-LESS test_primesense all', images=total,
                  scenes=reports, model_count=30, models=models, annotated_instances=target_instances,
                  tracking_runner_ready=False,
                  remaining=['verify viewpoint order and persistent instance association',
                             'normalize variable intrinsics for all five trackers without changing object geometry',
                             'convert PLY to OBJ preserving coordinates/units',
                             'implement symmetry-aware errors and a shared failure/reset rule',
                             'audit texture availability and model-feature branch activity'])
    save(output, result)
    print('Input structure audit saved: '+str(output), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['download', 'audit'])
    p.add_argument('--root', type=Path, default=Path(DEFAULT))
    p.add_argument('--output', type=Path)
    p.add_argument('--endpoint', help='Download origin; defaults to HF_ENDPOINT or https://huggingface.co')
    args = p.parse_args()
    if args.action == 'download':
        download(args.root, args.endpoint)
    else:
        audit(args.root, args.output or args.root/'tracking_input_audit.json')
