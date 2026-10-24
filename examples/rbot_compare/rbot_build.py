"""Bind separately built RBOT drivers to the existing frozen comparison sources."""
import argparse
import json
from pathlib import Path
import sys
from rbot_data import METHODS, sha256, write_json
from generate_adapters import generate

HERE = Path(__file__).resolve().parent


def old_identity(kit, method):
    sys.path.insert(0, str(kit))
    from provenance import verified_build
    return verified_build(kit, method)


def adapter_hashes():
    return {p.relative_to(HERE).as_posix(): sha256(p)
            for pattern in ['*.py', '*.sh', 'CMakeLists.txt', 'build_inputs/*.cpp'] for p in HERE.glob(pattern)}


def record(kit, method):
    build = HERE/'build'/method
    value = dict(method=method, binary_sha256=sha256(build/'evaluate_rbot_comparison'),
                 original_comparison_identity=old_identity(kit, method), adapter_sha256=adapter_hashes(),
                 cmake_cache_sha256=sha256(build/'CMakeCache.txt'), comparison_kit=str(Path(kit).resolve()))
    write_json(build/'build_identity.json', value)


def verified(kit, method):
    build = HERE/'build'/method
    value = json.loads((build/'build_identity.json').read_text(encoding='utf-8'))
    if (value['method'] != method or value['binary_sha256'] != sha256(build/'evaluate_rbot_comparison')
            or value['original_comparison_identity'] != old_identity(kit, method)
            or value['adapter_sha256'] != adapter_hashes()
            or value['cmake_cache_sha256'] != sha256(build/'CMakeCache.txt')
            or value['comparison_kit'] != str(Path(kit).resolve())):
        raise ValueError('RBOT build/source changed; rebuild before starting a new run: '+method)
    return value


def prepare(kit):
    kit = Path(kit).resolve()
    for method in METHODS:
        old_identity(kit, method)
    return generate(kit/'baseline.cpp', kit/'sources/current_m3t', HERE/'build_inputs')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'record', 'verify'])
    p.add_argument('--kit', type=Path, default=Path('/home/jyj/Projects/industrial-tracker-comparison'))
    p.add_argument('--method', choices=METHODS)
    args = p.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.kit), indent=2))
    elif not args.method:
        p.error('--method is required')
    elif args.action == 'record':
        record(args.kit, args.method)
    else:
        print(json.dumps(verified(args.kit, args.method), indent=2))
