"""Generate copies of the frozen evaluator; never patch the original algorithms."""
import argparse
import difflib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'support'))
if not (HERE/'support/rbot_data.py').exists():sys.path.insert(0,str(HERE.parent/'rbot_compare'))
from rbot_data import sha256,write_json


def replace_once(raw,old,new):
    if raw.count(old)!=1:raise ValueError('Frozen source pattern changed: '+old[:100])
    return raw.replace(old,new,1)


def video_only_compatibility(raw):
    """Keep the original Eigen in-place reductions when no CAD matching is used.

    The frozen extension materializes a full signed 6x6 product, whereas the
    original reduces directly into its lower triangle. Preserve the original
    operation ordering in the video-only branch; retain the CAD branch exactly.
    """
    marker='''    // Calculate gradient and hessian
    const Eigen::Matrix<float, 6, 1> gradient_summand{'''
    restored='''    // With CAD matching disabled, preserve the upstream video-only
    // Eigen in-place reductions (not a full negated temporary product).
    // Diagnostics above still run when fusion requires their statistics.
    if (!data_point.from_model_texture && !use_model_texture_matching_) {
      gradient_ -= (weight * diff.transpose()) * dx_dtheta;
      hessian_.triangularView<Eigen::Lower>() -=
          (weight * dx_dtheta.transpose()) * dx_dtheta;
      continue;
    }

'''+marker
    return replace_once(raw,marker,restored)


def generate(current,output):
    current=Path(current);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original_h=(current/'examples/rbot_evaluator.h').read_text(encoding='utf-8-sig')
    original_cpp=(current/'examples/rbot_evaluator.cpp').read_text(encoding='utf-8-sig')
    h=replace_once(original_h,'  void set_use_directional_temporal_prior(bool use_directional_temporal_prior);',
        '  void set_use_ablation_prediction_smoothing(bool enabled) { use_ablation_prediction_smoothing_ = enabled; }\n'
        '  void set_use_directional_temporal_prior(bool use_directional_temporal_prior);')
    h=replace_once(h,'  bool use_directional_temporal_prior_ = false;',
        '  bool use_directional_temporal_prior_ = false;\n  bool use_ablation_prediction_smoothing_ = true;')
    cpp=replace_once(original_cpp,'use_directional_temporal_prior_ &&\n                 smoothed_prediction_ready',
        'use_ablation_prediction_smoothing_ &&\n                 smoothed_prediction_ready')
    # These errors are offline diagnostics. They do not control the enabled P/T
    # paths, but the old wall clock included them even when no log was requested.
    cpp=replace_once(cpp,'    if (predicted_pose_valid) {','    if (collect_diagnostics_ && predicted_pose_valid) {')
    cpp=replace_once(cpp,'<< "input_conversion" << "uint16_to_uint8_fixed_1_over_257"',
        '<< "input_conversion" << (external_sequence_.first_image_index == 0 ? "native_uint8_bgr" : "uint16_to_uint8_fixed_1_over_257")')
    source=(current/'examples/evaluate_industrial_dataset.cpp').read_text(encoding='utf-8-sig')
    setters=source[source.index('    evaluator.set_external_sequence(input);'):source.index('    std::cout << "Industrial final:')]
    cad_begin=setters.index('      t->set_use_model_texture_disk(true,')
    cad_end=setters.index('      t->set_descriptor_distance_threshold',cad_begin)
    cad_code=setters[cad_begin:cad_end]
    setters=setters[:cad_begin]+'      if (cad) {\n'+cad_code+'      } else {\n        t->set_use_model_texture_disk(false);\n        t->DoNotUseModelTextureMatching();\n      }\n'+setters[cad_end:]
    for method,flag in [('noise_adaptive_filtering','noise'),('gaussian_noise_filter','noise'),
                        ('constant_velocity_prediction','prediction'),('reliability_weighted_fusion','fusion'),('directional_temporal_prior','prior')]:
        setters=replace_once(setters,'evaluator.set_use_'+method+'(true);','evaluator.set_use_'+method+'('+flag+');')
    cli=replace_once((HERE/'ablation_cli.cpp').read_text(encoding='utf-8-sig'),'    // @@FROZEN_SETTERS@@',setters)
    original_texture=(current/'src/texture_modality.cpp').read_text(encoding='utf-8-sig')
    texture=video_only_compatibility(original_texture)
    for name,raw in [('rbot_evaluator.h',h),('rbot_evaluator.cpp',cpp),('ablation.cpp',cli),('texture_modality.cpp',texture)]:
        with (output/name).open('w',encoding='utf-8',newline='\n') as stream:stream.write(raw)
    patch=''.join(difflib.unified_diff(original_h.splitlines(True),h.splitlines(True),fromfile='frozen/rbot_evaluator.h',tofile='generated/rbot_evaluator.h'))
    patch+=''.join(difflib.unified_diff(original_cpp.splitlines(True),cpp.splitlines(True),fromfile='frozen/rbot_evaluator.cpp',tofile='generated/rbot_evaluator.cpp'))
    (output/'evaluator_changes.patch').write_text(patch,encoding='utf-8')
    texture_patch=''.join(difflib.unified_diff(original_texture.splitlines(True),texture.splitlines(True),
        fromfile='frozen/src/texture_modality.cpp',tofile='generated/texture_modality.cpp'))
    (output/'video_only_arithmetic.patch').write_text(texture_patch,encoding='utf-8')
    write_json(output/'generation.json',dict(source={name:sha256(current/'examples'/name) for name in
        ['rbot_evaluator.h','rbot_evaluator.cpp','evaluate_industrial_dataset.cpp']},
        texture_source_sha256=sha256(current/'src/texture_modality.cpp'),
        compatibility='upstream in-place video gradient/Hessian when CAD matching is off; unchanged CAD-enabled source path',
        generated={p.name:sha256(p) for p in output.glob('*') if p.suffix in ['.cpp','.h','.patch']}))


def old(kit,method):
    sys.path.insert(0,str(kit))
    from provenance import verified_build
    return verified_build(kit,method)


def hashes():
    return {p.relative_to(HERE).as_posix():sha256(p) for pattern in
            ['ablation*.py','*.sh','CMakeLists.txt','expected_core.json','variants.json','support/*.py','build_inputs/*.cpp','build_inputs/*.h','build_inputs/*.patch'] for p in HERE.glob(pattern)}


def identity(kit):
    return dict(binary_sha256=sha256(HERE/'build/evaluate_ablation'),sources={m:old(kit,m) for m in ['ours','m3t_original_rgb']},
                helpers=hashes(),cmake_cache_sha256=sha256(HERE/'build/CMakeCache.txt'))


def verified(kit):
    recorded=json.loads((HERE/'build/identity.json').read_text())
    if recorded!=identity(kit):raise ValueError('Ablation build/source changed; rebuild for a new run')
    return recorded


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','record']);p.add_argument('--kit',type=Path,required=True);a=p.parse_args()
    if a.action=='generate':
        payload=HERE/'payload_manifest.json'
        if payload.exists():
            for name,value in json.loads(payload.read_text())['sha256'].items():
                if sha256(HERE/name)!=value:raise ValueError('Changed helper payload: '+name)
        origin=old(a.kit,'ours');old(a.kit,'m3t_original_rgb')
        for name,value in json.loads((HERE/'expected_core.json').read_text()) .items():
            if origin['source_sha256'].get('source/'+name)!=value:raise ValueError('Full core differs from the reviewed snapshot: '+name)
        generate(a.kit/'sources/current_m3t',HERE/'build_inputs')
    else:write_json(HERE/'build/identity.json',identity(a.kit))
