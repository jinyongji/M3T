"""Independent advanced-v1 calibration, pairing, pose and mask audit.

Run with Blender's bundled Python/NumPy or ordinary Python with NumPy.
Does not execute supplied scripts or rewrite dataset inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

NAME = "工业跟踪进阶数据集-v1.0-20261018"
SCENES = {"scene_01": [1, 2, 3], "scene_04": [10, 11, 12], "scene_05": [13, 14, 15]}
CONDITIONS = ["B00", "C01", "C02", "C03", "R01", "R02", "R03", "O01", "O02", "O03", "M01", "M02"]
TRAJECTORIES = ["t01", "t02", "t03"]
MODE = "camera_and_objects_motion"
K = [[1125, 0, 449.5], [0, 1125, 299.5], [0, 0, 1]]


def require(value, message):
    if not value:
        raise ValueError(message)


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    path.relative_to(root)
    require(path != root, "Empty input path")
    return path


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def check_se3(matrices, label):
    require(np.isfinite(matrices).all(), "Nonfinite "+label)
    require(np.max(np.abs(matrices[..., 3, :]-[0, 0, 0, 1])) < 2e-6, "Invalid homogeneous row: "+label)
    r = matrices[..., :3, :3]
    require(np.max(np.abs(np.swapaxes(r, -1, -2)@r-np.eye(3))) < 1e-5, "Nonrigid rotation: "+label)
    require(np.max(np.abs(np.linalg.det(r)-1)) < 1e-5, "Improper rotation: "+label)


def check_events(occlusion, truncation, focus, condition):
    require(np.isfinite(occlusion).all() and np.isfinite(truncation).all(), "Nonfinite visibility")
    require(np.min(occlusion) >= -1e-9 and np.max(occlusion) < 1-1e-9, "Fully occluded/invalid object")
    require(np.min(truncation) >= -1e-9 and np.max(truncation) <= 1/3+1e-6, "Object truncation exceeds one third")
    require(np.max(np.abs(truncation[:, focus])) < 1e-9, "Focus object is out of frame")
    result = {"max_truncation": float(truncation.max()), "focus_occlusion_max": float(occlusion[:, focus].max())}
    if condition.startswith("O"):
        desired = int(condition[-1])*.2
        plateau = np.r_[np.arange(110, 150), np.arange(310, 350)]
        outside = np.ones(500, bool); outside[100:160] = False; outside[300:360] = False
        values = occlusion[plateau, focus]
        require(np.max(np.abs(values-desired)) <= .05+1e-9, "Occlusion plateau outside target band")
        require(np.max(occlusion[outside, focus]) < .05, "Unscheduled focus occlusion")
        result.update(plateau_min=float(values.min()), plateau_max=float(values.max()),
                      non_event_max=float(occlusion[outside, focus].max()))
    return result


def audit(root, output, masks=True):
    root = Path(root).resolve(); data = root/"交付物"/NAME
    camera = json.loads((data/"cameras/camera.json").read_text(encoding="utf-8-sig"))
    for key, expected in {"width": 900, "height": 600, "fps": 30, "frames_per_sequence": 500,
                          "K": K, "translation_unit": "meter", "pose_direction": "p_camera = R @ p_object + t",
                          "conditions": CONDITIONS, "motion_modes": [MODE], "global_shutter_exposure_s": .0005}.items():
        require(camera.get(key) == expected, "Unexpected camera contract: "+key)
    require(np.array_equal(np.loadtxt(data/"cameras/K.txt"), K), "K.txt disagrees with camera.json")
    require(all(x == 0 for x in camera["distortion_coefficients"]), "Nonzero distortion")
    expected_ids = sorted(i for ids in SCENES.values() for i in ids)
    require(sorted(m["object_id"] for m in camera["models"]) == expected_ids and
            all(m["units"] == "meter" for m in camera["models"]), "Unexpected model IDs/units")
    with (data/"cameras/sequence_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 108 and len({r["sequence_id"] for r in rows}) == 108, "Need 108 unique sequences")
    require({(r["scene"], r["trajectory"], r["condition"]) for r in rows} ==
            {(s, t, c) for s in SCENES for t in TRAJECTORIES for c in CONDITIONS}, "Missing/unknown sequence")
    metadata_cache = {}; geometric = {}; poses = {}; manifest = {}; jobs = []
    max_pose_error = 0.
    for row in rows:
        s, t, c = (row[k] for k in ("scene", "trajectory", "condition"))
        ids = SCENES[s]; focus = int(t[-1])-1
        require(int(row["focus_object_id"]) == ids[focus], "Wrong focus object")
        require([int(v) for v in row["object_ids"].split(";")] == ids, "Index object IDs disagree")
        require((int(row["width"]), int(row["height"]), int(row["frames"]), int(row["fps"])) == (900, 600, 500, 30), "Index calibration mismatch")
        require(float(row["exposure_ms"]) == .5 and row["motion_mode"] == MODE, "Index exposure/motion mismatch")
        require(row["split"] == ("development" if t == "t01" else "heldout_trajectory"), "Unexpected pair split")
        require(row["pair_id"] == s+"_"+t and row["reference_sequence"] == "supp_"+s+"_"+t+"_B00", "Invalid pairing reference")
        relative = Path(s)/MODE/t/c
        require(row["rgb_path"] == (Path("images")/relative).as_posix() and
                row["poses_path"] == (Path("poses")/relative).as_posix() and
                row["video_path"] == (Path("videos")/relative).with_suffix(".mp4").as_posix(), "Index path does not match sequence")
        meta_file = data/"poses/metadata"/s/MODE/t/(c+".json")
        meta = json.loads(meta_file.read_text(encoding="utf-8-sig"))
        for key in row:
            if key in ("focus_object_id", "occluder_object_id", "frames", "fps", "width", "height"):
                require(int(row[key]) == meta[key], "Metadata/index integer mismatch: "+key)
            elif key == "exposure_ms":
                require(float(row[key]) == meta[key], "Metadata exposure mismatch")
            else:
                require(row[key] == meta[key], "Metadata/index mismatch: "+key)
        npz_path = local_path(data, row["geometry_metadata"])
        if npz_path not in metadata_cache:
            with np.load(npz_path, allow_pickle=False) as z:
                value = {key: z[key] for key in z.files}
            require(np.array_equal(value["ids"], ids) and int(value["focus_id"]) == ids[focus], "NPZ object IDs mismatch")
            require(np.array_equal(value["frame"], np.arange(1, 501)) and
                    np.allclose(value["time_seconds"], np.arange(500)/30, rtol=0, atol=1e-12), "Frame/time shift")
            require(value["object_to_world"].shape == (500, 3, 4, 4) and value["camera_to_world"].shape == (500, 4, 4), "World-pose shape mismatch")
            check_se3(value["object_to_world"], str(npz_path)); check_se3(value["camera_to_world"], str(npz_path))
            metadata_cache[npz_path] = value
        value = metadata_cache[npz_path]
        expected = np.linalg.inv(value["camera_to_world"])[:, None]@value["object_to_world"]
        truth = []
        for oid in ids:
            pose_file = local_path(data, row["poses_path"])/("poses_%06d.txt" % oid)
            p = np.loadtxt(pose_file)
            require(p.shape == (500, 12) and np.isfinite(p).all() and (p[:, 11] > 0).all(), "Invalid/truncated RBOT TXT: "+str(pose_file))
            truth.append(p)
            manifest[pose_file.relative_to(data).as_posix()] = hashlib.sha256(pose_file.read_bytes()).hexdigest()
            jobs.append(dict(row, object="obj_%06d" % oid, is_focus=oid == ids[focus],
                             evaluated_frames=499, ground_truth_file=pose_file.relative_to(data).as_posix()))
        truth = np.stack(truth, axis=1)
        flattened = np.concatenate([expected[..., :3, :3].reshape(500, 3, 9), expected[..., :3, 3]], axis=-1)
        error = float(np.max(np.abs(flattened-truth)))
        require(error < 1e-5, "TXT disagrees with world-matrix NPZ: "+row["sequence_id"])
        max_pose_error = max(max_pose_error, error); poses[s, t, c] = truth
        key = c if c.startswith(("O", "M")) else "B00"
        if (s, t, key) not in geometric:
            item = check_events(value["occlusion"], value["truncation"], focus, key)
            item["scene"] = s; item["trajectory"] = t; item["condition"] = key
            if masks:
                source = root/"中间结果文件/geometry_annotations"/s/t/(key+".npz")
                with np.load(source, allow_pickle=False) as z:
                    visible, amodal = z["visible"], z["amodal"]
                    require(np.array_equal(z["ids"], ids), "Mask object IDs mismatch")
                require(visible.shape == amodal.shape == (500, 3, 600, 113) and visible.dtype == amodal.dtype == np.uint8, "Mask calibration/shape mismatch")
                lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
                visible_area = np.empty((500, 3), dtype=np.int64); area = visible_area.copy()
                for first in range(0, 500, 10):
                    v, am = visible[first:first+10], amodal[first:first+10]
                    require(not np.any(v & ~am), "Visible mask is outside amodal mask")
                    require(not np.any(v[..., -1] & 15) and not np.any(am[..., -1] & 15), "Mask padding bits are set")
                    visible_area[first:first+10] = lut[v].sum((2, 3))
                    area[first:first+10] = lut[am].sum((2, 3))
                require(np.array_equal(area, value["area"]) and np.all(area > 0), "Mask projected areas disagree")
                require(np.max(np.abs(1-visible_area/area-value["occlusion"])) < 1e-12, "Mask-derived occlusion disagrees")
                del visible, amodal
            geometric[s, t, key] = item
        print("AUDITED", row["sequence_id"], flush=True)
    for s in SCENES:
        for t in TRAJECTORIES:
            reference = poses[s, t, "B00"]
            base = metadata_cache[local_path(data, "poses/metadata/"+s+"/"+MODE+"/"+t+"/B00.npz")]
            focus = int(t[-1])-1
            exposures = []
            for c in CONDITIONS:
                meta = json.loads((data/"poses/metadata"/s/MODE/t/(c+".json")).read_text(encoding="utf-8"))
                exposures.append(meta["exposure_compensation"])
                item = metadata_cache[local_path(data, meta["geometry_metadata"])]
                require(np.array_equal(base["camera_to_world"], item["camera_to_world"]), "Paired camera trajectories differ")
                if c.startswith(("C", "R")):
                    require(np.array_equal(reference, poses[s, t, c]), "C/R poses differ from B00")
                elif c.startswith("O"):
                    require(np.array_equal(reference[:, focus], poses[s, t, c][:, focus]), "O focus poses differ from B00")
                elif c.startswith("M"):
                    require(not np.allclose(reference, poses[s, t, c], rtol=0, atol=1e-5), "M poses copied from B00")
            require(len(set(exposures)) == 1, "Exposure compensation changes within a pair")
    index_file = Path(output).with_name("evaluation_index.json")
    save(index_file, {"schema_version": 1, "dataset": str(data), "conditions": CONDITIONS,
                      "input_policy": "RGB and CAD only; masks and future GT are offline evaluation aids",
                      "target_configurations": len(jobs), "focus_configurations": sum(x["is_focus"] for x in jobs),
                      "evaluated_frames_all_objects_per_method": len(jobs)*499,
                      "evaluated_frames_focus_per_method": sum(x["is_focus"] for x in jobs)*499, "configurations": jobs})
    report = {"status": "passed", "dataset": str(data), "resolution": [900, 600], "K": K,
              "rescaling_required": False, "sequences": len(rows), "pose_files": len(jobs), "pose_rows": len(jobs)*500,
              "independent_geometry_sets": len(geometric), "mask_arrays_audited": masks,
              "max_txt_npz_pose_error": max_pose_error, "maximum_truncation": max(x["max_truncation"] for x in geometric.values()),
              "geometry": list(geometric.values()), "pose_sha256": manifest, "evaluation_index": str(index_file),
              "scope": "input consistency and precomputed geometry; actual Blender scene evaluation and rendered outputs audited separately"}
    save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-masks", action="store_true")
    import sys
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    result = audit(args.root, args.output, not args.skip_masks)
    print("INPUT_AUDIT_PASSED", json.dumps({k: result[k] for k in ("sequences", "pose_rows", "resolution", "maximum_truncation")}), flush=True)
