"""Audit every pre-reset industrial pose; invalid estimates count as failures."""
import csv
import json
import math
import statistics
from pathlib import Path

from dataset import determinant, paths, read_poses


def orthogonal_rotation(matrix):
    matrix = list(matrix)
    for _ in range(30):
        a, b, c, d, e, f, g, h, i = matrix
        cofactor = [e*i-f*h, f*g-d*i, d*h-e*g, c*h-b*i, a*i-c*g, b*g-a*h,
                    b*f-c*e, c*d-a*f, a*e-b*d]
        det = a*cofactor[0]+b*cofactor[1]+c*cofactor[2]
        if not math.isfinite(det) or det <= 1e-12:
            raise ValueError("Singular/reflected estimated rotation")
        updated = [(x+y/det)/2 for x, y in zip(matrix, cofactor)]
        if max(abs(x-y) for x, y in zip(updated, matrix)) < 1e-12:
            return updated
        matrix = updated
    raise ValueError("Rotation polar projection did not converge")


def pose(row, prefix):
    return ([float(row["%s_r%d%d" % (prefix, r, c)]) for r in range(3) for c in range(3)]
            + [float(row["%s_t%s_m" % (prefix, axis)]) for axis in "xyz"])


def analyze(directory, root, config, image_count):
    directory = Path(directory)
    files = list((directory / "results/poses").glob("poses_*.csv"))
    if len(files) != 1:
        raise ValueError("Expected one complete pose CSV: %s" % directory)
    path = files[0]
    metadata_path = path.with_name(path.name.replace("poses_", "metadata_", 1)).with_suffix(".json")
    meta = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    expected = {"schema_version": 1, "dataset_kind": "industrial", "object": config["object"],
                "sequence": "__".join(config[k] for k in ("scene", "motion", "condition")),
                "translation_unit": "m", "model_unit_in_meter": 1, "ground_truth_translation_scale": 1,
                "pose_convention": "p_camera = R * p_body + t", "initial_image_index": 1,
                "first_image_index": 2, "last_image_index": image_count, "frame_count": image_count-1,
                "reset_protocol": "industrial_selected_output_reset",
                "estimate_capture": "final_selected_output_before_gt_reset",
                "input_conversion": "uint16_to_uint8_fixed_1_over_257"}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError("Unexpected metadata %s in %s" % (key, metadata_path))
    intrinsics = {"fx": 1125, "fy": 1125, "cx": 449.5, "cy": 299.5, "width": 900, "height": 600}
    if meta.get("intrinsics") != intrinsics:
        raise ValueError("Exported intrinsics disagree with dataset")
    for key, expected_value in (("translation_threshold_m", .05), ("rotation_threshold_rad", math.radians(5))):
        if abs(meta[key]-expected_value) > 1e-7:
            raise ValueError("Unexpected native success threshold")
    _, gt_file, _ = paths(root, config)
    truth = read_poses(gt_file)
    initial = meta["initial_gt_pose_camera_row_major"]
    expected_initial = [truth[0][r*3+c] if c < 3 else truth[0][9+r]
                        for r in range(3) for c in range(4)] + [0, 0, 0, 1]
    if (len(initial) != 16 or not all(math.isfinite(v) for v in initial)
            or any(abs(a-b) > 2e-6 for a, b in zip(initial, expected_initial))):
        raise ValueError("Initial pose disagrees with image 000001 GT")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != image_count-1:
        raise ValueError("Incomplete CSV: %s" % path)
    segment = native = recomputed = invalid = 0
    translations, rotations, cycles = [], [], []
    disagreements, native_anomalies = [], []
    for frame, row in enumerate(rows):
        image = frame+2
        if (int(row["frame"]) != frame or int(row["image_index"]) != image
                or row["image_file"] != "%06d.png" % image or int(row["segment_id"]) != segment
                or row["estimate_source"] != "single"):
            raise ValueError("Misaligned frame/segment: %s row %d" % (path, frame))
        for key in ("tracking_success", "reset_primary_after_frame", "reset_secondary_after_frame", "pose_finite"):
            if row[key] not in ("0", "1"):
                raise ValueError("Invalid flag %s" % key)
        success, reset = int(row["tracking_success"]), int(row["reset_primary_after_frame"])
        if reset != 1-success or row["reset_secondary_after_frame"] != "0":
            raise ValueError("Native single tracker reset protocol violated")
        estimate, gt = pose(row, "est"), pose(row, "gt")
        if not all(math.isfinite(v) for v in gt) or any(abs(a-b) > 2e-6 for a, b in zip(gt, truth[image-1])):
            raise ValueError("Exported GT differs from the source pose row")
        finite = all(math.isfinite(v) for v in estimate)
        if int(finite) != int(row["pose_finite"]):
            raise ValueError("pose_finite does not describe the exported matrix")
        valid = finite and determinant(estimate[:9]) > 1e-8
        if valid:
            er, gr = orthogonal_rotation(estimate[:9]), orthogonal_rotation(gt[:9])
            translation = math.sqrt(sum((a-b)**2 for a, b in zip(estimate[9:], gt[9:])))
            rotation = math.acos(max(-1, min(1, (sum(a*b for a, b in zip(er, gr))-1)/2)))
            translations.append(translation)
            rotations.append(math.degrees(rotation))
            passed = translation <= .05 and rotation <= math.radians(5)
        else:
            invalid += 1
            passed = False
            if success:
                raise ValueError("Invalid estimated pose was counted successful")
        nt, nr = float(row["translation_error_m"]), float(row["rotation_error_rad"])
        if not math.isfinite(nt) or not math.isfinite(nr):
            native_anomalies.append(image)
            if success:
                raise ValueError("Non-finite native error was counted successful")
        cycle = float(row["cycle_us"])
        if not math.isfinite(cycle) or cycle < 0:
            raise ValueError("Invalid cycle time")
        cycles.append(cycle)
        if success != int(passed):
            disagreements.append(image)
        native += success
        recomputed += int(passed)
        segment += reset
    overlay = directory / "results/overlays/initial_gt_image_1.png"
    if not overlay.is_file() or overlay.stat().st_size < 100:
        raise ValueError("Initial GT overlay is missing: %s" % overlay)
    return dict(config, frames=len(rows), successes=native, recomputed_successes=recomputed,
                success_percent=100*native/len(rows), recomputed_success_percent=100*recomputed/len(rows),
                reset_count=segment, invalid_estimate_count=invalid, error_statistics_frame_count=len(translations),
                translation_mean_m=statistics.mean(translations) if translations else None,
                rotation_mean_deg=statistics.mean(rotations) if rotations else None,
                cycle_mean_us=statistics.mean(cycles),
                success_classification_disagreement_count=len(disagreements),
                success_classification_disagreement_images=disagreements,
                nonfinite_native_error_images=native_anomalies, pose_csv=str(path), initial_gt_overlay=str(overlay))


def aggregate(configs):
    frames = sum(c["frames"] for c in configs)
    return {"configurations": len(configs), "frames": frames,
            "success_percent": 100*sum(c["successes"] for c in configs)/frames,
            "recomputed_success_percent": 100*sum(c["recomputed_successes"] for c in configs)/frames,
            "invalid_estimate_count": sum(c["invalid_estimate_count"] for c in configs),
            "success_classification_disagreement_count": sum(c["success_classification_disagreement_count"] for c in configs),
            "cycle_mean_us": sum(c["cycle_mean_us"]*c["frames"] for c in configs)/frames}
