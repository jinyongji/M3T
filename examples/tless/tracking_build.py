"""Generate a separate driver with frozen DLR engines and a camera injection shim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tracking_data import METHODS,save,sha

HERE=Path(__file__).resolve().parent


def old(kit,method):
    sys.path.insert(0,str(kit))
    from provenance import verified_build
    return verified_build(kit,method)


def generate(baseline,current,output):
    base=Path(baseline).read_text(encoding='utf-8-sig')
    prefix=base[:base.index('struct Input {')]
    prefix=prefix.replace('constexpr const char *kMethod = "m3t_original_rgb";',
        '#if defined(BENCH_OURS)\nconstexpr const char *kMethod = "ours";\n#else\nconstexpr const char *kMethod = "m3t_original_rgb";\n#endif')
    prefix+='\n#if defined(BENCH_OURS)\n#include <m3t/loader_camera.h>\n#include <m3t/textured_renderer.h>\n#endif\n#include <deque>\n'
    engine=base[base.index('class Engine {'):base.index('\nstruct Row {')]
    engine=engine.replace('#if defined(BENCH_M3T)','#if defined(BENCH_M3T) || defined(BENCH_OURS)')
    engine=engine.replace('explicit Engine(const Input &in):camera(std::make_shared<IndustrialCamera>(in))',
                          'explicit Engine(const Input &in):camera(std::make_shared<IndustrialCamera>(in)),input_(in)')
    engine=engine.replace('body->set_body2world_pose(pose);','''body->set_body2world_pose(pose);
#if defined(BENCH_OURS)
    history_.clear();history_.push_back(pose);link->ClearDirectionalTemporalPrior();
#endif''')
    engine=engine.replace('Require(body->SetUp(),"Body setup failed");','''Require(body->SetUp(),"Body setup failed");
#if defined(BENCH_OURS)
    Require(!body->has_texture(),"This protocol is frozen for the supplied textureless CAD; use a separate experiment for textured models");
#endif''')
    engine=engine.replace('link->AddModality(region);','''
#if defined(BENCH_OURS)
    link->set_use_reliability_weighted_fusion(true);
#endif
    link->AddModality(region);''')
    engine=engine.replace('Times t;const auto start=Clock::now();','''Times t;const auto start=Clock::now();
#if defined(BENCH_OURS)
    Predict();
#endif''')
    engine=engine.replace('region->set_n_unoccluded_iterations(0);','''region->set_n_unoccluded_iterations(0);
#if defined(BENCH_OURS)
    region->set_collect_diagnostics(true);
#endif''')
    ours=Path(current)/'examples/evaluate_industrial_dataset.cpp'
    source=ours.read_text(encoding='utf-8-sig')
    setters=source.split('evaluator.set_texture_modality_setter([&](auto t) {',1)[1].split('\n    });',1)[0]
    setters=setters.replace('t->','texture->').replace('args.at("--disk-model")','input_.disk_model').replace('object +','input_.object +')
    marker='  void ConfigureTexture() {'
    engine=engine.replace(marker,marker+'\n#if defined(BENCH_OURS)\n'+setters+'\n    texture->set_collect_diagnostics(true);\n#else')
    target='texture->set_modeled_occlusion_radius(0);texture->set_modeled_occlusion_threshold(.03f);\n  }'
    if engine.count(target)!=1:raise ValueError('Frozen texture template changed')
    engine=engine.replace(target,target.replace('\n  }','\n#endif\n  }'))
    engine=engine.replace(' private:\n','''
#if defined(BENCH_OURS)
  void Accept(){history_.push_back(Estimated());while(history_.size()>3)history_.pop_front();}
#endif
 private:
  const Input &input_;
#if defined(BENCH_OURS)
  std::deque<Pose> history_;
  void Predict(){
    if(history_.size()<2){link->ClearDirectionalTemporalPrior();return;}
    const auto last=history_.back();Pose prediction=last*history_[history_.size()-2].inverse()*last;
    if(history_.size()==3){
      const Pose a=history_[0].inverse()*history_[1],b=history_[1].inverse()*history_[2];
      const Eigen::AngleAxisf ar(a.rotation().matrix()),br(b.rotation().matrix());
      const Eigen::Vector3f rv=.5f*(ar.angle()*ar.axis()+br.angle()*br.axis());
      Pose delta=Pose::Identity();delta.translate(.5f*(a.translation()+b.translation()));
      if(rv.norm()>std::numeric_limits<float>::min())delta.rotate(Eigen::AngleAxisf(rv.norm(),rv.normalized()));
      prediction=last*delta;
    }
    body->set_body2world_pose(prediction);link->SetDirectionalTemporalPrior(prediction);
  }
#endif
''',1)
    driver=(HERE/'tracking_driver.cpp').read_text(encoding='utf-8-sig')
    code=prefix+driver.replace('// @@ENGINE@@',engine)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'tless.cpp').write_text(code,encoding='utf-8')
    header=(Path(current)/'include/m3t/loader_camera.h').read_text(encoding='utf-8-sig')
    injection='''  // Adapter-only decoded-frame input; reuse the original filtering and raw-image accessor.
  void SetAdapterFrame(const cv::Mat &value) {
    image_ = value; unfiltered_image_ = value; FilterImageIfRequested();
  }
'''
    header=header.replace('  bool SetUp() override;','  bool SetUp() override;\n'+injection,1)
    override=output/'override/m3t';override.mkdir(parents=True,exist_ok=True)
    (override/'loader_camera.h').write_text(header,encoding='utf-8')
    save(output/'generation.json',dict(baseline_sha256=sha(baseline),full_setters_sha256=sha(ours),
                                     original_loader_header_sha256=sha(Path(current)/'include/m3t/loader_camera.h'),
                                     camera_shim='one public inline setter added in an override header; original engine files untouched',
                                     generated_sha256=sha(output/'tless.cpp'),header_sha256=sha(override/'loader_camera.h')))


def helpers():
    return {p.relative_to(HERE).as_posix():sha(p) for pattern in ['tracking_*.py','tracking_*.cpp','run_tracking.sh','build_tracking.sh',
                                                             'CMakeLists.txt','expected_sources.json','build_inputs/tless.cpp',
                                                             'build_inputs/override/m3t/loader_camera.h'] for p in HERE.glob(pattern)}


def identity(kit,method):
    folder=HERE/'build'/method
    return dict(method=method,binary_sha256=sha(folder/'evaluate_tless'),source=old(kit,method),
                helpers=helpers(),cmake_cache_sha256=sha(folder/'CMakeCache.txt'))


def verified(kit,method):
    expected=json.loads((HERE/'build'/method/'identity.json').read_text())
    if expected!=identity(kit,method):raise ValueError('T-LESS build/source changed; rebuild before a new run')
    return expected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','record']);p.add_argument('--kit',type=Path,required=True);p.add_argument('--method',choices=METHODS)
    a=p.parse_args()
    if a.action=='generate':
        manifest=HERE/'payload_manifest.json'
        if manifest.is_file():
            for name,expected in json.loads(manifest.read_text(encoding='utf-8'))['sha256'].items():
                if sha(HERE/name)!=expected:raise ValueError('T-LESS helper payload differs: '+name)
        for m in METHODS:old(a.kit,m)
        generate(a.kit/'baseline.cpp',a.kit/'sources/current_m3t',HERE/'build_inputs')
    else:save(HERE/'build'/a.method/'identity.json',identity(a.kit,a.method))
