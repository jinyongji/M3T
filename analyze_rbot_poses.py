"""Recompute pose errors from RBOT exports using only Python's standard library."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def pose_values(row, prefix):
    rotation = [float(row[f"{prefix}_r{r}{c}"]) for r in range(3) for c in range(3)]
    translation = [float(row[f"{prefix}_t{axis}_m"]) for axis in "xyz"]
    if not all(math.isfinite(v) for v in rotation + translation):
        raise ValueError("Non-finite pose; invalid frames must not be silently dropped")
    return rotation, translation


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def orthogonal_rotation(matrix):
    """Polar factor of a positive-determinant 3x3 block, without NumPy.

    M3T stores Eigen::Affine poses; its rotation() uses SVD to remove tiny
    scale/shear drift. Newton's polar iteration gives the same rotation in
    double precision for these nonsingular, orientation-preserving matrices.
    The exported raw matrix is never overwritten.
    """
    matrix = list(matrix)
    for _ in range(20):
        a, b, c, d, e, f, g, h, i = matrix
        cofactor = [e*i-f*h, f*g-d*i, d*h-e*g,
                    c*h-b*i, a*i-c*g, b*g-a*h,
                    b*f-c*e, c*d-a*f, a*e-b*d]
        determinant = a*cofactor[0] + b*cofactor[1] + c*cofactor[2]
        if not math.isfinite(determinant) or determinant <= 1e-12:
            raise ValueError("Rotation block is singular or reflected; cannot interpret as an SE(3) pose")
        updated = [(value + inverse_t/determinant)/2 for value, inverse_t in zip(matrix, cofactor)]
        if max(abs(a-b) for a, b in zip(updated, matrix)) < 1e-12:
            return updated
        matrix = updated
    raise ValueError("Rotation polar projection did not converge")


def analyze_file(path, translation_threshold, rotation_threshold_deg):
    metadata_path = path.with_name(path.name.replace("poses_", "metadata_", 1)).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if (metadata["schema_version"] != 1 or metadata["translation_unit"] != "m"
            or metadata["pose_convention"] != "p_camera = R * p_body + t"):
        raise ValueError(f"Unsupported pose schema: {metadata_path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata["frame_count"] or not rows:
        raise ValueError(f"Incomplete pose file: {path}")
    translations, rotations = [], []
    native_successes = successes = resets = segment = 0
    max_translation_difference = max_rotation_difference = 0.0
    for frame, row in enumerate(rows):
        image_index = metadata["first_image_index"] + frame
        if (int(row["frame"]) != frame or int(row["image_index"]) != image_index
                or row["image_file"] != f'{metadata["sequence"]}{image_index:04d}.png'
                or int(row["segment_id"]) != segment or row["pose_finite"] != "1"):
            raise ValueError(f"Invalid frame identity/segment/pose: {path}, row {frame}")
        for key in ("tracking_success", "reset_primary_after_frame", "reset_secondary_after_frame"):
            if row[key] not in ("0", "1"):
                raise ValueError(f"Invalid flag {key}: {path}, row {frame}")
        primary, secondary = int(row["reset_primary_after_frame"]), int(row["reset_secondary_after_frame"])
        if row["estimate_source"] not in ("single", "disk", "orb"):
            raise ValueError(f"Unknown estimate source: {path}, row {frame}")
        if row["estimate_source"] == "single" and (secondary or primary != 1 - int(row["tracking_success"])):
            raise ValueError(f"Invalid single-tracker reset: {path}, row {frame}")
        er, et = pose_values(row, "est")
        gr, gt = pose_values(row, "gt")
        translation = math.sqrt(sum((a - b) ** 2 for a, b in zip(et, gt)))
        er, gr = orthogonal_rotation(er), orthogonal_rotation(gr)
        # trace(R_est.T @ R_gt) is the Frobenius inner product.
        cosine = (sum(a * b for a, b in zip(er, gr)) - 1) / 2
        rotation = math.acos(max(-1.0, min(1.0, cosine)))
        translations.append(translation)
        rotations.append(math.degrees(rotation))
        native_t, native_r = float(row["translation_error_m"]), float(row["rotation_error_rad"])
        if not math.isfinite(native_t) or not math.isfinite(native_r):
            raise ValueError(f"Non-finite native errors: {path}, row {frame}")
        max_translation_difference = max(max_translation_difference, abs(translation - native_t))
        max_rotation_difference = max(max_rotation_difference, abs(rotation - native_r))
        successes += translation <= translation_threshold and math.degrees(rotation) <= rotation_threshold_deg
        native_successes += int(row["tracking_success"])
        if primary or secondary:
            resets += 1
            segment += 1
    if int(rows[-1]["image_index"]) != metadata["last_image_index"]:
        raise ValueError(f"Last frame does not match metadata: {path}")
    return {
        "file": str(path.resolve()), "object": metadata["object"], "sequence": metadata["sequence"],
        "frames": len(rows), "reset_protocol": metadata["reset_protocol"],
        "native_success_percent": 100 * native_successes / len(rows),
        "recomputed_success_percent": 100 * successes / len(rows),
        "translation_mean_m": statistics.mean(translations),
        "translation_rmse_m": math.sqrt(statistics.mean(v * v for v in translations)),
        "translation_p95_m": percentile(translations, 95),
        "rotation_mean_deg": statistics.mean(rotations),
        "rotation_rmse_deg": math.sqrt(statistics.mean(v * v for v in rotations)),
        "rotation_p95_deg": percentile(rotations, 95), "frames_followed_by_reset": resets,
        "max_difference_from_native_translation_m": max_translation_difference,
        "max_difference_from_native_rotation_rad": max_rotation_difference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Run directory, results/poses directory, or one pose CSV")
    parser.add_argument("--translation-threshold-m", type=float, default=0.05)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--output", type=Path, help="Optional JSON path; without it only print the report")
    args = parser.parse_args()
    for threshold in (args.translation_threshold_m, args.rotation_threshold_deg):
        if not math.isfinite(threshold) or threshold < 0:
            parser.error("Thresholds must be finite and non-negative")
    if args.input.is_file():
        paths = [args.input]
    else:
        paths = []
        for directory in (args.input / "results" / "poses", args.input / "poses", args.input):
            paths = sorted(directory.glob("poses_*.csv"))
            if paths:
                break
    if not paths:
        parser.error("No poses_*.csv files found; old runs need to be evaluated again")
    try:
        if args.input.is_dir():
            # A complete CSV from an interrupted 72-configuration job is not a
            # complete benchmark. A deliberate single-CSV analysis is allowed.
            for directory in (args.input, args.input.parent, args.input.parent.parent):
                manifest_path = directory / "run_manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                    if manifest.get("Status") != "complete" or manifest.get("ExpectedConfigurations") != len(paths):
                        raise ValueError("Run is incomplete or configuration count differs from its manifest")
                    break
        configurations = [analyze_file(path, args.translation_threshold_m, args.rotation_threshold_deg) for path in paths]
        if len({c["reset_protocol"] for c in configurations}) != 1:
            raise ValueError("Do not aggregate different reset protocols; analyze them separately")
        report = {
            "translation_threshold_m": args.translation_threshold_m,
            "rotation_threshold_deg": args.rotation_threshold_deg,
            "rotation_error_method": "polar_projection_then_geodesic_angle_double_precision",
            "note": "Metrics use the recorded reset trajectory. Changing thresholds here does not rerun tracking or change its resets.",
            "configurations": configurations,
            "configuration_count": len(configurations),
            "total_frames": sum(c["frames"] for c in configurations),
            "configuration_macro_success_percent": statistics.mean(c["recomputed_success_percent"] for c in configurations),
        }
        rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            # Avoid silently destroying an earlier analysis or an input file.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        # ASCII console JSON also works with legacy Windows code pages. The
        # optional report file above is UTF-8 with readable Unicode paths.
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Pose analysis failed: {error}\n")


if __name__ == "__main__":
    main()
