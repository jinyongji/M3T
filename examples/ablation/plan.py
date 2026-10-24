"""Write an auditable ablation design; this is intentionally not a tracker runner."""
import argparse
import hashlib
import json
from pathlib import Path

MODULES = ['cad_disk', 'reliability_complement_fusion', 'prediction_initialization',
           'directional_prior', 'region_noise_filter']
DESIGN = [
    ('A0', 'compatibility_all_off', [0, 0, 0, 0, 0]),
    ('A1', 'full', [1, 1, 1, 1, 1]),
    ('A2', 'without_cad_constraint', [0, 1, 1, 1, 1]),
    ('A3', 'without_fusion_module', [1, 0, 1, 1, 1]),
    ('A4', 'without_prediction_initialization', [1, 1, 0, 1, 1]),
    ('A5', 'without_directional_prior', [1, 1, 1, 0, 1]),
    ('A6', 'without_region_noise_filter', [1, 1, 1, 1, 0]),
    ('A7', 'without_all_motion_modules', [1, 1, 0, 0, 1]),
]
CORE_FILES = ['src/link.cpp', 'src/texture_modality.cpp', 'src/loader_camera.cpp',
              'src/body.cpp', 'src/renderer_geometry.cpp', 'src/textured_renderer.cpp',
              'src/disk_feature_extractor.cpp', 'examples/rbot_evaluator.cpp',
              'examples/evaluate_industrial_dataset.cpp']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(current, original, result):
    frozen = result['identity']['builds']['ours']['original_comparison_identity']['source_sha256']
    source_audit = []
    for name in CORE_FILES:
        raw = (current/name).read_bytes()
        prior = original/name
        pinned = frozen.get('source/'+name)
        source_audit.append(dict(file=name, local_sha256=sha(raw),
                                 local_sha256_lf=sha(raw.replace(b'\r\n', b'\n')),
                                 original_sha256_lf=sha(prior.read_bytes().replace(b'\r\n', b'\n')) if prior.is_file() else None,
                                 rbot_frozen_sha256=pinned,
                                 matches_rbot_frozen=(sha(raw) == pinned) if pinned else None))
    matched = {r['file']:r['matches_rbot_frozen'] for r in source_audit}
    for name in ['src/link.cpp','src/texture_modality.cpp','src/loader_camera.cpp',
                 'examples/rbot_evaluator.cpp','examples/evaluate_industrial_dataset.cpp']:
        if matched[name] is not True:
            raise ValueError('Full algorithm core differs from the exported RBOT snapshot: '+name)
    variants = [dict(id=i, name=n, modules=dict(zip(MODULES, map(bool, state)))) for i,n,state in DESIGN]
    datasets = [dict(id='rbot', gpu_lane='A', configurations=72, evaluated_frames_per_configuration=1000,
                     input='/home/jyj/autodl-tmp/RBOT-dataset'),
                dict(id='industrial_base', gpu_lane='B', configurations=150, evaluated_frames_per_configuration=499,
                     input='/home/jyj/autodl-tmp/industrial-render-900x600/dataset/交付物/工业物体跟踪数据集-v0.8-900x600-20261016'),
                dict(id='industrial_advanced', gpu_lane='C', configurations=324, evaluated_frames_per_configuration=499,
                     input='/home/jyj/autodl-tmp/industrial-render-advanced-v1/dataset/交付物/工业跟踪进阶数据集-v1.0-20261018')]
    for d in datasets:
        d.update(gpu_uuid=None, evaluated_frames_per_variant=d['configurations']*d['evaluated_frames_per_configuration'],
                 jobs=len(variants)*d['configurations'], accuracy_repeats=1,
                 output_root='/home/jyj/autodl-tmp/M3T-ablation-runs/'+d['id'],
                 cache_root='/home/jyj/autodl-tmp/M3T-ablation-cache/'+d['id'])
    return dict(schema_version=1, date='2026-10-23', status='design_reviewable_implementation_pending',
                runner_ready=False, no_server_jobs_launched=True,
                purpose='Controlled component attribution, not five-method reruns or new accuracy claims',
                source=dict(current=str(current), original=str(original),
                            rbot_ours_binary_sha256=result['identity']['builds']['ours']['binary_sha256'],
                            rbot_icg_plus_binary_sha256=result['identity']['builds']['m3t_original_rgb']['binary_sha256'],
                            files=source_audit),
                common=dict(input='RGB only', region_modality=True, video_orb=True,
                            sensor_depth=False, corr_iterations=7, update_iterations=2,
                            rotation_regularization=1000, translation_regularization=30000,
                            disk_input_size=128, disk_ransac_threshold_px=3,
                            threshold_translation_m=0.05, threshold_rotation_deg=5,
                            initialization='first image GT; exclude initialization image from accuracy',
                            reset='count failure and save estimated pose before resetting to that frame GT',
                            warmup_timing_frames=10, primary_timing='compute_wall_us',
                            no_known_occluder_ground_truth=True,
                            diagnostics='separate explanatory runs; exclude verbose diagnostics from formal timing'),
                core_variants=variants,
                additional_priority_control=dict(id='A8', name='compatibility_plus_region_noise_filter',
                                                 modules=dict(zip(MODULES,[False,False,False,False,True]))),
                switch_requirements=dict(
                    cad_disabled=['set_use_model_texture_disk(false)', 'DoNotUseModelTextureMatching()',
                                  'do not create CAD textured renderer or DISK session; preserve video ORB'],
                    fusion_disabled=['use existing false branch; removes both reliability and information complement projection',
                                     'directional prior must still follow its independent switch'],
                    prediction_initialization_disabled=['continue computing the prior target if T is enabled',
                                                        'do not write predicted pose as optimization initialization'],
                    directional_prior_disabled=['preserve Full two-velocity prediction smoothing',
                                                'disable only application of directional prior'],
                    noise_disabled=['disable adaptive filtering; keep the same image conversion and raw ORB inputs'],
                    compatibility=['all new active modules off, same Region + video ORB and optimization parameters',
                                   'compare with external original ICG+ before attributing gains']),
                implementation_needed=[
                    'configurable independent driver; existing final wrappers hard-code all modules enabled',
                    'separate prediction smoothing from directional-prior flag without changing Full behavior',
                    'new provenance, binary/config hashing, per-frame pose/timing/reset audit and resume',
                    'separate cache roots; existing RBOT coordinator holds a global industrial cache lock',
                    'optional independent scalar-reliability and directional-complement switches for mechanism study'],
                gpu_execution=dict(lanes='one complete dataset per GPU; variants sequential within each lane',
                                   selection='inspect current load, then bind an actual UUID',
                                   accuracy_parallel=True, formal_timing_parallel=False,
                                   formal_timing_repeats_minimum=3,
                                   note='CPU/llvmpipe, memory and disk remain shared across GPUs'),
                datasets=datasets,
                total_core_jobs=sum(d['jobs'] for d in datasets),
                total_core_evaluated_frames=len(variants)*sum(d['evaluated_frames_per_variant'] for d in datasets),
                with_additional_noise_control=dict(jobs=(len(variants)+1)*sum(d['configurations'] for d in datasets),
                                                   evaluated_frames=(len(variants)+1)*sum(d['evaluated_frames_per_variant'] for d in datasets)),
                continuous_tracking_requirements=[
                    'separate protocol from the GT-reset benchmark',
                    'no GT-based reset, prediction-history update/gating, recovery trigger or online confidence',
                    'GT may be used for common initialization and offline evaluation only'],
                no_ablation_results_available=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--rbot-summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.rbot_summary.read_bytes()
    result = build(a.current, a.original, json.loads(raw.decode('utf-8-sig')))
    result['source']['rbot_summary_sha256'] = sha(raw)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(a.output), runner_ready=result['runner_ready'],
                          variants=len(result['core_variants']), jobs=result['total_core_jobs'],
                          evaluated_frames=result['total_core_evaluated_frames']),ensure_ascii=False))
