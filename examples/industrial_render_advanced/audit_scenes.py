"""Evaluate every integer-frame Blender pose and calibration, without rendering."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import bpy
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from audit_inputs import NAME, SCENES, TRAJECTORIES, CONDITIONS, MODE, K, require, save


def read_bounds(path):
    vertices = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
    points = np.array(vertices)
    center = (points.min(0)+points.max(0))/2
    return center, (points.max(0)-points.min(0))/2, float(np.linalg.norm(points-center, axis=1).max())


def obb_gap(world, bounds):
    """Conservative clearance: either OBB separation or enclosing spheres proves it.

    Overlapping loose boxes do not themselves imply triangle intersection.
    """
    smallest = math.inf
    for first, second in ((0, 1), (0, 2), (1, 2)):
        a, b = world[:, first], world[:, second]
        ra, rb = a[:, :3, :3], b[:, :3, :3]
        ca, ha, radius_a = bounds[first]; cb, hb, radius_b = bounds[second]
        delta = (np.einsum("nij,j->ni", rb, cb)+b[:, :3, 3] -
                 np.einsum("nij,j->ni", ra, ca)-a[:, :3, 3])
        axes_a, axes_b = ra.transpose(0, 2, 1), rb.transpose(0, 2, 1)
        cross = np.cross(axes_a[:, :, None], axes_b[:, None, :]).reshape(-1, 9, 3)
        axes = np.concatenate([axes_a, axes_b, cross], axis=1)
        norms = np.linalg.norm(axes, axis=2)
        valid = norms > 1e-10
        axes = axes/np.maximum(norms[..., None], 1e-10)
        distance = np.abs(np.einsum("nki,ni->nk", axes, delta))
        extent_a = np.einsum("nkj,j->nk", np.abs(np.einsum("nki,nij->nkj", axes, ra)), ha)
        extent_b = np.einsum("nkj,j->nk", np.abs(np.einsum("nki,nij->nkj", axes, rb)), hb)
        gap = np.where(valid, distance-extent_a-extent_b, -math.inf).max(1)
        gap = np.maximum(gap, np.linalg.norm(delta, axis=1)-radius_a-radius_b)
        smallest = min(smallest, float(gap.min()))
    return smallest


def movement(world):
    translation = np.linalg.norm(np.diff(world[:, :3, 3], axis=0), axis=1)
    rotations = world[:, :3, :3]
    cosine = (np.einsum("nij,nij->n", rotations[:-1], rotations[1:])-1)/2
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return {"translation_mean_mm_per_frame": float(translation.mean()*1000),
            "translation_max_mm_per_frame": float(translation.max()*1000),
            "rotation_mean_deg_per_frame": float(angle.mean()), "rotation_max_deg_per_frame": float(angle.max()),
            "translation_xyz_span_m": np.ptp(world[:, :3, 3], axis=0).tolist()}


def audit(root, output, only_scene=None, stride=1):
    root = Path(root).resolve(); data = root/"交付物"/NAME
    require(tuple(bpy.app.version) == (4, 5, 4), "Use the bundled Blender 4.5.4")
    flip = np.diag([1., -1., -1., 1.]); records = []; began = time.monotonic()
    frame_ids = list(range(1, 501, stride))
    if frame_ids[-1] != 500: frame_ids.append(500)
    for scene, ids in SCENES.items():
        if only_scene and scene != only_scene: continue
        blend = data/"scenes"/scene/(scene+"-advanced-v1.0-20261018.blend")
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        require(not bpy.data.libraries, "Unpacked external library dependency")
        require(not [im.name for im in bpy.data.images if im.source == "FILE" and not im.packed_file], "Unpacked texture dependency")
        require(set(bpy.data.scenes.keys()) == {t+"_"+c for t in TRAJECTORIES for c in CONDITIONS}, "Expected 36 Blender scenes")
        bounds = [read_bounds(data/"objects"/("obj_%06d" % oid)/("obj_%06d.obj" % oid)) for oid in ids]
        for trajectory in TRAJECTORIES:
            for condition in CONDITIONS:
                sc = bpy.data.scenes[trajectory+"_"+condition]; bpy.context.window.scene = sc
                for obj in sc.objects: obj.update_tag(refresh={"OBJECT"})
                sc.frame_set(1); bpy.context.view_layer.update()
                require((sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage) == (900, 600, 100), "Unexpected actual render resolution")
                require(sc.render.fps == 30 and sc.render.fps_base == 1, "Unexpected actual framerate")
                require(sc.render.pixel_aspect_x == sc.render.pixel_aspect_y == 1, "Non-square pixels")
                require(sc.render.use_motion_blur and sc.render.motion_blur_position == "CENTER" and
                        abs(sc.render.motion_blur_shutter-.015) < 1e-7, "Exposure is not 0.5ms centered")
                objects = sorted([o for o in sc.objects if "object_id" in o], key=lambda o: int(o["object_id"]))
                require([int(o["object_id"]) for o in objects] == ids, "Scene object IDs mismatch")
                for obj, bound in zip(objects, bounds):
                    native = np.array([v.co[:] for v in obj.data.vertices])
                    center, half, radius = bound
                    require(np.abs(native.min(0)-(center-half)).max() < 1e-7 and
                            np.abs(native.max(0)-(center+half)).max() < 1e-7, "Scene geometry bounds differ from OBJ")
                    require(not obj.modifiers and not obj.data.shape_keys, "Animated/deformed geometry needs a separate mesh audit")
                relative = Path(scene)/MODE/trajectory/condition
                meta = json.loads((data/"poses/metadata"/relative.with_suffix(".json")).read_text(encoding="utf-8"))
                require(abs(sc.view_settings.exposure-meta["exposure_compensation"]) < 1e-6, "Scene exposure disagrees with metadata")
                truth = np.stack([np.loadtxt(data/"poses"/relative/("poses_%06d.txt" % oid)) for oid in ids], axis=1)
                with np.load(data/meta["geometry_metadata"], allow_pickle=False) as z:
                    expected_camera, expected_world = z["camera_to_world"], z["object_to_world"]
                worlds = []; cameras = []; maximum = 0.; world_error = 0.; intrinsics_error = 0.
                for frame in frame_ids:
                    sc.frame_set(frame); bpy.context.view_layer.update(); dg = bpy.context.evaluated_depsgraph_get()
                    camera = np.asarray(sc.camera.evaluated_get(dg).matrix_world).copy()@flip
                    world = np.array([np.asarray(obj.evaluated_get(dg).matrix_world).copy() for obj in objects])
                    pose = np.linalg.inv(camera)[None]@world
                    values = np.concatenate([pose[:, :3, :3].reshape(3, 9), pose[:, :3, 3]], axis=1)
                    maximum = max(maximum, float(np.abs(values-truth[frame-1]).max()))
                    world_error = max(world_error, float(np.abs(world-expected_world[frame-1]).max()),
                                      float(np.abs(camera-expected_camera[frame-1]).max()))
                    projection = np.asarray(sc.camera.calc_matrix_camera(dg, x=900, y=600,
                                              scale_x=sc.render.pixel_aspect_x, scale_y=sc.render.pixel_aspect_y))
                    actual_k = [[projection[0, 0]*450, 0, (1-projection[0, 2])*450-.5],
                                [0, projection[1, 1]*300, (1+projection[1, 2])*300-.5], [0, 0, 1]]
                    intrinsics_error = max(intrinsics_error, float(np.abs(np.asarray(actual_k)-K).max()))
                    worlds.append(world); cameras.append(camera)
                require(maximum < 1e-5 and world_error < 1e-5, "Actual Blender poses differ from TXT/NPZ")
                require(intrinsics_error < 1e-3, "Actual Blender projection differs from K")
                world = np.array(worlds); camera = np.array(cameras)
                gap = obb_gap(world, bounds)
                require(gap > 0, "Neither enclosing spheres nor OBBs certify separation at a sampled frame")
                actors = {str(oid): movement(world[:, i]) for i, oid in enumerate(ids)} if stride == 1 else {}
                if stride == 1:
                    require(all(np.max(np.ptp(world[:, i, :3, 3], axis=0)) > .01 for i in range(3)), "Object world trajectory is static")
                item = {"scene": scene, "trajectory": trajectory, "condition": condition,
                        "frames_checked": len(frame_ids), "pose_error_max": maximum, "world_matrix_error_max": world_error,
                        "intrinsics_error_max": intrinsics_error, "integer_frame_enclosure_gap_min_m": gap,
                        "camera": movement(camera) if stride == 1 else {}, "objects": actors}
                records.append(item)
                save(output, {"status": "running", "completed_sequences": len(records), "sequences": records})
                print("SCENE_AUDITED", scene, trajectory, condition, "pose", maximum, "gap", gap,
                      "seconds", round(time.monotonic()-began, 1), flush=True)
    report = {"status": "passed", "scope": "all integer frames" if stride == 1 else "sampled integer frames",
              "sequences_checked": len(records), "total_frame_checks": sum(r["frames_checked"] for r in records),
              "max_pose_error": max(r["pose_error_max"] for r in records),
              "max_intrinsics_error": max(r["intrinsics_error_max"] for r in records),
              "minimum_integer_frame_enclosure_gap_m": min(r["integer_frame_enclosure_gap_min_m"] for r in records),
              "continuous_collision_scope": "integer-frame OBB or enclosing-sphere separation checked; supplied 240Hz and continuous certificates are separate evidence, not recomputed here",
              "seconds": time.monotonic()-began, "sequences": records}
    save(output, report); print("BLENDER_SCENE_AUDIT_PASSED", len(records), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene", choices=SCENES); p.add_argument("--stride", type=int, default=1)
    args = p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:])
    require(1 <= args.stride <= 100, "Stride must be 1..100")
    audit(args.root, args.output, args.scene, args.stride)
