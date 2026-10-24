"""Build an independent RGB experiment from the frozen T-LESS driver."""
import argparse
import difflib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent

def modules(tless):
    sys.path.insert(0,str(tless))
    import tracking_build,tracking_data
    return tracking_build,tracking_data

def once(raw,old,new):
    if raw.count(old)!=1:raise ValueError('Frozen adapter changed: '+old[:90])
    return raw.replace(old,new,1)

def generate(baseline,current,output,tless):
    old,data=modules(tless);output=Path(output)
    old.generate(baseline,current,output)
    path=output/'tless.cpp';original=path.read_text(encoding='utf-8-sig');code=original
    code=once(code,'struct View {','#include "rgb_modality.h"\n\nstruct View {')
    code=once(code,'  int frames=72;engine::Intrinsics','  std::string variant; bool video_verify=false,visibility_consensus=false,region_control=false;\n  int frames=72;engine::Intrinsics')
    code=once(code,'Require(args.size()==7,"Unknown argument");Input in;',
        'Require(args.size()==8 && args.count("--variant"),"Expected RGB variant");Input in;\n'
        '  in.variant=args.at("--variant");\n'
        '  Require(in.variant=="B0"||in.variant=="B1"||in.variant=="B2"||in.variant=="B3"||in.variant=="B4","Unknown RGB variant");\n'
        '  in.video_verify=in.variant=="B1"||in.variant=="B3";\n'
        '  in.visibility_consensus=in.variant=="B2"||in.variant=="B3";\n'
        '  in.region_control=in.variant=="B4";')
    code=once(code,'std::make_shared<engine::TextureModality>("texture",body,camera,silhouette)',
                    'std::make_shared<tless_rgb::VerifiedTexture>("texture",body,camera,silhouette)')
    code=once(code,'std::shared_ptr<engine::TextureModality> texture;',
                    'std::shared_ptr<tless_rgb::VerifiedTexture> texture;')
    code=once(code,'    texture->set_collect_diagnostics(true);',
        '    texture->set_collect_diagnostics(true);\n'
        '    texture->verify_video=input_.video_verify;texture->region_control=input_.region_control;')
    code=once(code,'    region->set_collect_diagnostics(true);',
        '    region->set_collect_diagnostics(true);\n'
        '    region->set_use_visibility_consensus(input_.visibility_consensus);\n'
        '    region->set_visibility_local_memory(input_.visibility_consensus);')
    code=once(code,'  Pose Estimated()const{return body->body2world_pose();}',
        '  Pose Estimated()const{return body->body2world_pose();}\n'
        '  int VideoCandidates()const{return texture->candidates;}\n'
        '  int VideoRetained()const{return texture->retained;}\n'
        '  bool VideoVerified()const{return texture->verified;}\n'
        '  float VisibilityEvidence()const{return region->visibility_evidence();}\n'
        '  float VisibilityActivation()const{return region->visibility_activation();}')
    code=once(code,'results_us\\n";',
        'results_us,rgb_video_candidates,rgb_video_retained,rgb_video_verified,rgb_visibility_evidence,rgb_visibility_activation\\n";')
    code=once(code,'const Pose estimate=tracker.Estimated(),gt=input.poses[i];',
        'const Pose estimate=tracker.Estimated(),gt=input.poses[i];\n'
        '      const int rgb_candidates=tracker.VideoCandidates(),rgb_retained=tracker.VideoRetained();\n'
        '      const bool rgb_verified=tracker.VideoVerified();\n'
        '      const float rgb_evidence=tracker.VisibilityEvidence(),rgb_activation=tracker.VisibilityActivation();')
    code=once(code,"<<times.results<<'\\n';",
        "<<times.results<<','<<rgb_candidates<<','<<rgb_retained<<','<<rgb_verified<<','<<rgb_evidence<<','<<rgb_activation<<'\\n';")
    code=once(code,'meta<<"method"<<kMethod',
        'meta<<"rgb_variant"<<input.variant<<"rgb_video_verification"<<int(input.video_verify)<<"rgb_visibility_consensus"<<int(input.visibility_consensus)<<"rgb_region_control"<<int(input.region_control);\n    meta<<"method"<<kMethod')
    path.write_text(code,encoding='utf-8')
    header=(Path(current)/'include/m3t/texture_modality.h').read_text(encoding='utf-8-sig')
    shim=once(header,'  bool SetUp() override;',
        '  bool SetUp() override;\n  auto &RgbAdapterPoints() { return data_points_; }')
    (output/'override/m3t/texture_modality.h').write_text(shim,encoding='utf-8')
    (output/'rgb_changes.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='frozen/tless.cpp',tofile='rgb/tless.cpp')),encoding='utf-8')
    data.save(output/'rgb_generation.json',dict(frozen_driver_sha256=data.sha(Path(tless)/'tracking_driver.cpp'),
        original_texture_header_sha256=data.sha(Path(current)/'include/m3t/texture_modality.h'),
        policy='No base implementation/layout changes; runtime flags; original metrics, frames, reset and timing retained',
        generated_sha256=data.sha(path),header_sha256=data.sha(output/'override/m3t/texture_modality.h')))

def identity(kit,tless):
    old,data=modules(tless)
    return dict(binary_sha256=data.sha(HERE/'build/evaluate_tless_rgb'),reference=old.verified(kit,'ours'),
        helpers={p.relative_to(HERE).as_posix():data.sha(p) for pattern in ['rgb_*.py','rgb_*.h','rgb_*.cpp','*.sh','CMakeLists.txt','variants.json','build_inputs/*.cpp','build_inputs/*.patch','build_inputs/override/m3t/*.h'] for p in HERE.glob(pattern)},
        cmake_cache_sha256=data.sha(HERE/'build/CMakeCache.txt'))

def verified(kit,tless):
    _,data=modules(tless);record=data.load(HERE/'build/identity.json')
    if record!=identity(kit,tless):raise ValueError('RGB build/source changed; start a new version')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','record'])
    p.add_argument('--kit',type=Path,default=Path('/home/jyj/Projects/industrial-tracker-comparison'))
    p.add_argument('--tless-kit',type=Path,default=Path('/home/jyj/Projects/tless-tracker-comparison'));a=p.parse_args()
    old,data=modules(a.tless_kit)
    if a.action=='generate':
        old.verified(a.kit,'ours')
        if (HERE/'payload_manifest.json').exists():
            for name,digest in data.load(HERE/'payload_manifest.json')['sha256'].items():
                if data.sha(HERE/name)!=digest:raise ValueError('Changed RGB payload: '+name)
        generate(a.kit/'baseline.cpp',a.kit/'sources/current_m3t',HERE/'build_inputs',a.tless_kit)
    else:data.save(HERE/'build/identity.json',identity(a.kit,a.tless_kit))
