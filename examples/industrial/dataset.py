"""Strict input contract for the rendered industrial v0.8 900x600 dataset.

Only standard-library dependencies; never rewrite source images/models/poses.
"""
import hashlib
import json
import math
import struct
from pathlib import Path

SCENES = ["scene_%02d" % i for i in range(1, 6)]
MOTIONS = ["camera_motion", "camera_and_objects_motion"]
CONDITIONS = ["01_normal", "02_low_light", "03_hard_glare", "04_fast_motion", "05_sensor_noise"]
DISK_SHA256 = "5ff1ecfd9a31009a12d0e73ec76b476291756fe101b0417e3d0cc90a0dd42648"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def determinant(r):
    a, b, c, d, e, f, g, h, i = r
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)


def read_camera(root, allow_motion_subset=False):
    path = Path(root) / "cameras/camera.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {"width": 900, "height": 600, "frames_per_sequence": 500,
                "translation_unit": "meter", "pose_direction": "p_camera = R @ p_object + t",
                "K": [[1125.0, 0, 449.5], [0, 1125.0, 299.5], [0, 0, 1]],
                "conditions": CONDITIONS}
    modes = data.get("motion_modes")
    if (not isinstance(modes, list) or not modes or len(modes) != len(set(modes))
            or not set(modes) <= set(MOTIONS) or (not allow_motion_subset and modes != MOTIONS)):
        raise ValueError("Unexpected motion modes in %s" % path)
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError("Unexpected camera contract %s in %s" % (key, path))
    distortion = data.get("distortion_coefficients")
    if not isinstance(distortion, list) or not distortion or any(v != 0 for v in distortion):
        raise ValueError("Expected zero distortion")
    models = data.get("models", [])
    if (len(models) != 15 or sorted(m.get("object_id", 0) for m in models) != list(range(1, 16))
            or any(m.get("units") != "meter" for m in models)):
        raise ValueError("Expected 15 unique metre-valued models")
    return data


def read_poses(path, expected_rows=500):
    poses = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 12 or not all(math.isfinite(v) for v in values):
                raise ValueError("Expected 12 finite pose values: %s:%d" % (path, number))
            r = values[:9]
            orth_error = max(abs(sum(r[k*3+i]*r[k*3+j] for k in range(3)) - int(i == j))
                             for i in range(3) for j in range(3))
            if abs(determinant(r)-1) > 1e-3 or orth_error > 1e-3:
                raise ValueError("GT rotation is not SO(3): %s:%d" % (path, number))
            poses.append(values)
    if len(poses) != expected_rows:
        raise ValueError("Expected %d poses, got %d: %s" % (expected_rows, len(poses), path))
    return poses


def configurations(scenes=None, motions=None, conditions=None, objects=None):
    scenes, motions, conditions = scenes or SCENES, motions or MOTIONS, conditions or CONDITIONS
    for selected, allowed in ((scenes, SCENES), (motions, MOTIONS), (conditions, CONDITIONS)):
        if len(selected) != len(set(selected)) or not set(selected) <= set(allowed):
            raise ValueError("Unknown or duplicate selection: %s" % selected)
    allowed_objects = {"obj_%06d" % i for i in range(1, 16)}
    if objects and (len(objects) != len(set(objects)) or not set(objects) <= allowed_objects):
        raise ValueError("Unknown or duplicate objects")
    result = []
    for scene in scenes:
        scene_id = int(scene[-2:])
        for motion in motions:
            for condition in conditions:
                for object_id in range(scene_id*3-2, scene_id*3+1):
                    obj = "obj_%06d" % object_id
                    if objects and obj not in objects:
                        continue
                    result.append({"scene": scene, "motion": motion, "condition": condition, "object": obj,
                                   "id": "__".join((scene, motion, condition, obj))})
    if not result:
        raise ValueError("Selection contains no target configurations")
    return result


def paths(root, config):
    root = Path(root)
    relative = Path(config["scene"]) / config["motion"] / config["condition"]
    obj = config["object"]
    return (root / "images" / relative, root / "poses" / relative / ("poses_%s.txt" % obj[4:]),
            root / "objects" / obj / (obj + ".obj"))


def png_header(path):
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR"
            or struct.unpack(">I", header[8:12])[0] != 13):
        raise ValueError("Invalid PNG IHDR: %s" % path)
    width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    if (width, height, bits, color, compression, filtering, interlace) != (900, 600, 16, 2, 0, 0, 0):
        raise ValueError("Expected 900x600 16-bit RGB non-interlaced PNG: %s" % path)
    if Path(path).stat().st_size <= 33:
        raise ValueError("Truncated PNG: %s" % path)
    return header


def preflight(root, configs, image_count, allow_motion_subset=False):
    if not 2 <= image_count <= 500:
        raise ValueError("Image count includes initialization and must be 2..500")
    root = Path(root).resolve()
    camera = read_camera(root, allow_motion_subset)
    if not {c["motion"] for c in configs} <= set(camera["motion_modes"]):
        raise ValueError("Selected motion is not declared in the camera metadata")
    fingerprints = {"cameras/camera.json": sha256(root / "cameras/camera.json")}
    images_checked = set()
    models_checked = set()
    for config in configs:
        images, pose_file, model = paths(root, config)
        read_poses(pose_file)
        fingerprints[pose_file.relative_to(root).as_posix()] = sha256(pose_file)
        if model not in models_checked:
            # OBJ, MTL and the actual diffuse map consumed by Body/RendererGeometry.
            # This algorithm uses one diffuse texture; reject ambiguous multi-map models.
            material = model.with_suffix(".mtl")
            textures = []
            for line in material.read_text(encoding="utf-8-sig").splitlines():
                if line.strip().startswith("map_Kd "):
                    relative = line.strip()[7:].strip()
                    texture = (model.parent / relative).resolve()
                    try:
                        texture.relative_to(model.parent.resolve())
                    except ValueError:
                        raise ValueError("Texture escapes object directory: %s" % material)
                    textures.append(texture)
            if len(set(textures)) != 1:
                raise ValueError("Current CAD texture renderer requires one diffuse map: %s" % material)
            for asset in [model, material] + textures:
                fingerprints[asset.relative_to(root).as_posix()] = sha256(asset)
            models_checked.add(model)
        for index in range(1, image_count + 1):
            path = images / ("%06d.png" % index)
            if path in images_checked:
                continue
            header = png_header(path)
            stat = path.stat()
            # Deliberately avoid rereading tens of GB of image pixels. Actual decoding
            # is checked by LoaderColorCamera while evaluating; record this scope.
            fingerprints[path.relative_to(root).as_posix()] = {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ihdr_sha256": hashlib.sha256(header).hexdigest()}
            images_checked.add(path)
    encoded = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"camera": camera, "fingerprint": hashlib.sha256(encoded).hexdigest(), "files": fingerprints,
            "image_validation": "PNG IHDR + size + mtime; full decode at evaluation time",
            "unique_images_checked": len(images_checked), "pose_files_checked": len(configs),
            "target_configurations": len(configs), "evaluated_frames": len(configs)*(image_count-1)}
