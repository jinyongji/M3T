"""Audit every RBOT pre-reset estimate, its image alignment and compute timing."""
import csv
import json
import math
from pathlib import Path
import statistics
from rbot_data import determinant, INTRINSICS

TIMES = ['cycle_us', 'decode_us', 'preprocessing_us', 'tracking_wall_us', 'compute_wall_us',
         'metric_us', 'reset_us', 'frame_wall_us', 'before_camera_us', 'correspondences_us',
         'gradient_hessian_us', 'optimization_us', 'results_us']


def orthogonal_rotation(matrix):
    matrix = list(matrix)
    for _ in range(30):
        a, b, c, d, e, f, g, h, i = matrix
        cofactor = [e*i-f*h, f*g-d*i, d*h-e*g, c*h-b*i, a*i-c*g, b*g-a*h,
                    b*f-c*e, c*d-a*f, a*e-b*d]
        det = a*cofactor[0]+b*cofactor[1]+c*cofactor[2]
        if not math.isfinite(det) or det <= 1e-12:
            raise ValueError('Singular/reflected estimated rotation')
        updated = [(x+y/det)/2 for x, y in zip(matrix, cofactor)]
        if max(abs(x-y) for x, y in zip(updated, matrix)) < 1e-12:
            return updated
        matrix = updated
    raise ValueError('Polar projection did not converge')


def pose(row, prefix):
    return ([float(row['%s_r%d%d' % (prefix, r, c)]) for r in range(3) for c in range(3)]
            + [float(row['%s_t%s_m' % (prefix, axis)]) for axis in 'xyz'])


def timing_rows(path, warmup):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(TIMES) <= set(reader.fieldnames or []):
            raise ValueError('Missing per-frame compute/decode columns; rerun the old cycle-only evaluation')
        rows = [{k: float(r[k]) for k in TIMES} for r in reader]
    if not 0 <= warmup < len(rows):
        raise ValueError('Warmup must leave timed frames')
    for r in rows:
        if any(not math.isfinite(v) or v < 0 for v in r.values()):
            raise ValueError('Nonfinite/negative timing')
        close = lambda a, b: abs(a-b) <= max(2, abs(a)*1e-4)
        if not close(r['compute_wall_us'], r['tracking_wall_us']-r['decode_us']):
            raise ValueError('Compute/decode timing decomposition disagrees')
        if not close(r['frame_wall_us'], r['tracking_wall_us']+r['metric_us']+r['reset_us']):
            raise ValueError('Frame timing decomposition disagrees')
        if r['preprocessing_us'] > r['compute_wall_us']+2:
            raise ValueError('Preprocessing exceeds compute time')
    return rows[warmup:]


def analyze(directory, config, truth, image_count):
    files = sorted((Path(directory)/'results/poses').glob('poses_*.csv'))
    if len(files) != 1:
        raise ValueError('Expected exactly one pose CSV: ' + str(directory))
    path = files[0]
    meta_file = path.with_name(path.name.replace('poses_', 'metadata_', 1)).with_suffix('.json')
    meta = json.loads(meta_file.read_text(encoding='utf-8-sig'))
    for key, expected in dict(schema_version=1, object=config['body'], sequence=config['sequence'],
                              translation_unit='m',
                              first_image_index=1, last_image_index=image_count-1,
                              frame_count=image_count-1, estimate_capture='final_selected_output_before_gt_reset').items():
        if meta.get(key) != expected:
            raise ValueError('Unexpected RBOT metadata ' + key)
    unit = meta.get('model_unit_in_meter', float('inf'))
    if not math.isfinite(unit) or abs(unit-.001) > 1e-9:
        raise ValueError('Incorrect RBOT model millimetre-to-metre scale')
    for key, value in INTRINSICS.items():
        if not math.isfinite(meta['intrinsics'][key]) or abs(meta['intrinsics'][key]-value) > 1e-4:
            raise ValueError('Incorrect RBOT intrinsics: ' + key)
    for key, value in [('translation_threshold_m', .05), ('rotation_threshold_rad', math.radians(5))]:
        if not math.isfinite(meta[key]) or abs(meta[key]-value) > 1e-7:
            raise ValueError('Incorrect success threshold')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != image_count-1:
        raise ValueError('Incomplete RBOT pose export')
    native = recomputed = invalid = segment = 0
    disagreements, translations, rotations, cycles = [], [], [], []
    for frame, row in enumerate(rows):
        image = frame+1
        if (int(row['frame']) != frame or int(row['image_index']) != image
                or row['image_file'] != '%s%04d.png' % (config['sequence'], image)
                or int(row['segment_id']) != segment or row['estimate_source'] != 'single'):
            raise ValueError('RBOT image/segment alignment error')
        for key in ['tracking_success', 'pose_finite', 'reset_primary_after_frame', 'reset_secondary_after_frame']:
            if row[key] not in ['0', '1']:
                raise ValueError('Invalid boolean flag ' + key)
        passed, reset = int(row['tracking_success']), int(row['reset_primary_after_frame'])
        if reset != 1-passed or row['reset_secondary_after_frame'] != '0':
            raise ValueError('Failure must be counted before the single target GT reset')
        e, gt = pose(row, 'est'), pose(row, 'gt')
        if not all(math.isfinite(x) for x in gt) or any(abs(a-b) > 2e-6 for a, b in zip(gt, truth[image])):
            raise ValueError('Exported GT differs from RBOT poses_first.txt')
        finite = all(math.isfinite(x) for x in e)
        if int(row['pose_finite']) != int(finite):
            raise ValueError('Incorrect pose_finite flag')
        valid = finite and determinant(e[:9]) > 1e-8
        if valid:
            er, gr = orthogonal_rotation(e[:9]), orthogonal_rotation(gt[:9])
            te = math.sqrt(sum((a-b)**2 for a, b in zip(e[9:], gt[9:])))
            re = math.acos(max(-1, min(1, (sum(a*b for a, b in zip(er, gr))-1)/2)))
            translations.append(te); rotations.append(math.degrees(re))
            audit_pass = te <= .05 and re <= math.radians(5)
        else:
            invalid += 1; audit_pass = False
            if passed:
                raise ValueError('Invalid estimate counted as success')
        if not all(math.isfinite(float(row[k])) for k in ['translation_error_m', 'rotation_error_rad']) and passed:
            raise ValueError('Nonfinite native error counted as success')
        cycles.append(float(row['cycle_us']))
        if passed != int(audit_pass):
            disagreements.append(image)
        native += passed; recomputed += int(audit_pass); segment += reset
    timing_rows(path, 0)
    return dict(config, frames=len(rows), successes=native, recomputed_successes=recomputed,
                success_percent=100*native/len(rows), recomputed_success_percent=100*recomputed/len(rows),
                reset_count=segment, invalid_estimate_count=invalid,
                success_classification_disagreement_count=len(disagreements),
                success_classification_disagreement_images=disagreements,
                translation_mean_m=statistics.mean(translations) if translations else None,
                rotation_mean_deg=statistics.mean(rotations) if rotations else None,
                cycle_mean_us=statistics.mean(cycles), pose_csv=str(path))
