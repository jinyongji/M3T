"""Generate separate RBOT drivers from the validated industrial engine adapters.

Only I/O, units and frame indexing change; source algorithm files stay untouched.
The generated files live in this kit's build_inputs directory, never the old kit.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent

COMMON_INPUT = r'''
struct Input {
  fs::path root,cache,output,images,mesh,truth;
  std::string object,sequence;
  int frames=1001;
  engine::Intrinsics intrinsics{650.048f,647.183f,323.828f,256.823f,640,512};
  std::vector<Pose> poses;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<std::array<int,3>> triangles;
  float diameter=0,legacy_model_diameter=0;
};
Input ReadInput(int argc,char **argv) {
  std::map<std::string,std::string> args;
  const std::vector<std::string> keys{"--dataset","--cache","--output","--body","--sequence","--frames","--poses"};
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];
    Require(std::find(keys.begin(),keys.end(),key)!=keys.end() && !args.count(key) && i+1<argc,"Invalid argument: "+key);
    args[key]=argv[++i];
  }
  for(const auto &key:keys)Require(args.count(key),"Missing "+key);
  Input in;in.root=args["--dataset"];in.cache=args["--cache"];in.output=args["--output"];
  in.object=args["--body"];in.sequence=args["--sequence"];in.truth=args["--poses"];
  for(const auto &part:{in.object,in.sequence})Require(!part.empty() && part!="." && part!=".." && part.find_first_of("/\\")==std::string::npos,"Invalid component");
  size_t consumed=0;in.frames=std::stoi(args["--frames"],&consumed);
  Require(consumed==args["--frames"].size() && in.frames>=2 && in.frames<=1001,"Frames must be 2..1001 including image 0000");
  in.images=in.root/in.object/"frames";in.mesh=in.root/in.object/(in.object+".obj");
  std::ifstream poses{in.truth};Require(bool(poses),"Cannot read normalized RBOT poses");
  std::string line;
  while(std::getline(poses,line)) {
    if(line.find_first_not_of(" \t\r")==std::string::npos)continue;
    std::istringstream row{line};row.imbue(std::locale::classic());Pose p=Pose::Identity();
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)Require(bool(row>>p.matrix()(r,c)),"Invalid pose rotation");
    for(int r=0;r<3;++r)Require(bool(row>>p.matrix()(r,3)),"Invalid pose translation");
    std::string extra;Require(!(row>>extra),"Extra pose fields");
    Require(p.matrix().allFinite() && std::abs(p.linear().determinant()-1)<1e-3 &&
      (p.linear().transpose()*p.linear()-Eigen::Matrix3f::Identity()).cwiseAbs().maxCoeff()<1e-3,"Invalid GT");
    in.poses.push_back(p);
  }
  Require(in.poses.size()==1001,"RBOT poses must have all 1001 rows");
  std::ifstream mesh{in.mesh};Require(bool(mesh),"Cannot read OBJ");
  while(std::getline(mesh,line)) {
    std::istringstream row{line};std::string kind;row>>kind;
    if(kind=="v") {
      Eigen::Vector3f p;Require(bool(row>>p.x()>>p.y()>>p.z()) && p.allFinite(),"Invalid OBJ vertex");
      p*=.001f;in.vertices.push_back(p);in.diameter=std::max(in.diameter,2*p.norm());
    } else if(kind=="f") {
      std::vector<int> face;std::string token;
      while(row>>token) {int n=std::stoi(token.substr(0,token.find('/')));face.push_back(n>0?n-1:int(in.vertices.size())+n);}
      for(size_t i=1;i+1<face.size();++i)in.triangles.push_back({face[0],face[i],face[i+1]});
    }
  }
  Require(in.diameter>0 && in.diameter<1.6f,"RBOT model outside fixed model sphere");
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  in.legacy_model_diameter=LegacyModelDiameter(in.diameter);
#endif
  for(const auto &t:in.triangles)for(int i:t)Require(i>=0 && i<int(in.vertices.size()),"Invalid triangle");
  fs::create_directories(in.cache);fs::create_directories(in.output/"results/poses");return in;
}
'''

CAMERA = r'''
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
using CameraBase=engine::Camera;
#else
using CameraBase=engine::ColorCamera;
#endif
class IndustrialCamera:public CameraBase {
 public:
  explicit IndustrialCamera(const Input &in)
#if defined(BENCH_RBGT)
    :CameraBase(),input_(in) {name_="camera";intrinsics_=in.intrinsics;}
#else
    :CameraBase("camera"),input_(in) {intrinsics_=in.intrinsics;}
#endif
  bool SetUp()
#if !defined(BENCH_RBGT)
    override
#endif
  {
    if(index_>0)return true;
#if defined(BENCH_RBGT)
    initialized_=true;
#else
    set_up_=true;
#endif
    return Load();
  }
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  bool UpdateImage() override {return Load();}
#else
  bool UpdateImage(bool synchronized) override {return Load();}
#endif
  double decode_us=0,preprocess_us=0;
 private:
  bool Load() {
    std::ostringstream name;name<<input_.sequence<<std::setw(4)<<std::setfill('0')<<index_<<".png";
    const auto start=Clock::now();image_=cv::imread((input_.images/name.str()).string(),cv::IMREAD_UNCHANGED);decode_us=Micros(start);
    Require(!image_.empty() && image_.rows==512 && image_.cols==640 && image_.type()==CV_8UC3,"Expected 640x512 uint8 RBOT RGB: "+name.str());
    ++index_;return true;
  }
  const Input &input_;int index_=0;
};
'''

SAVE = r'''
void Save(const Input &in,const std::vector<Row> &rows,double setup_us,const cv::Mat &initial) {
  const auto directory=in.output/"results/poses";const auto title=in.sequence+"_"+in.object;
  std::ofstream out{directory/("poses_"+title+".csv")};out.imbue(std::locale::classic());out<<std::setprecision(17);
  out<<"frame,image_index,image_file,segment_id,estimate_source,pose_finite,reset_primary_after_frame,reset_secondary_after_frame,tracking_success,translation_error_m,rotation_error_rad,cycle_us";
  for(const auto *prefix:{"est","gt"}) {for(int r=0;r<3;++r)for(int c=0;c<3;++c)out<<','<<prefix<<"_r"<<r<<c;out<<','<<prefix<<"_tx_m,"<<prefix<<"_ty_m,"<<prefix<<"_tz_m";}
  out<<",decode_us,preprocessing_us,tracking_wall_us,compute_wall_us,metric_us,reset_us,frame_wall_us,before_camera_us,correspondences_us,gradient_hessian_us,optimization_us,results_us\n";
  int segment=0;
  for(size_t i=0;i<rows.size();++i) {
    const auto &r=rows[i];out<<i<<','<<i+1<<','<<in.sequence<<std::setw(4)<<std::setfill('0')<<i+1<<".png,"<<segment<<",single,"<<int(r.estimate.matrix().allFinite())<<','<<int(!r.success)<<",0,"<<int(r.success)<<','<<r.translation<<','<<r.rotation<<','<<r.times.Native();
    for(const auto *p:{&r.estimate,&r.truth}) {for(int y=0;y<3;++y)for(int x=0;x<3;++x)out<<','<<p->matrix()(y,x);for(int y=0;y<3;++y)out<<','<<p->translation()(y);}
    Require(r.times.wall+2>=r.decode_us,"Decode exceeds tracking wall");
    out<<','<<r.decode_us<<','<<r.preprocess_us<<','<<r.times.wall<<','<<std::max(0.0,r.times.wall-r.decode_us)<<','<<r.metric_us<<','<<r.reset_us<<','<<r.times.wall+r.metric_us+r.reset_us
       <<','<<r.times.before<<','<<r.times.correspondences<<','<<r.times.gradient<<','<<r.times.optimization<<','<<r.times.results<<'\n';
    segment+=!r.success;
  }
  out.close();Require(!out.fail(),"Pose CSV write failed");
  cv::FileStorage meta{"",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON};
  meta<<"schema_version"<<1<<"dataset_kind"<<"rbot"<<"method"<<kMethod<<"object"<<in.object<<"sequence"<<in.sequence
      <<"translation_unit"<<"m"<<"model_unit_in_meter"<<.001<<"ground_truth_translation_scale"<<1.0
      <<"pose_convention"<<"p_camera = R * p_body + t"<<"first_image_index"<<1<<"last_image_index"<<in.frames-1<<"frame_count"<<int(rows.size())
      <<"estimate_capture"<<"final_selected_output_before_gt_reset"<<"translation_threshold_m"<<.05<<"rotation_threshold_rad"<<double(5*engine::kPi/180)
      <<"intrinsics"<<"{"<<"fx"<<double(in.intrinsics.fu)<<"fy"<<double(in.intrinsics.fv)<<"cx"<<double(in.intrinsics.ppu)<<"cy"<<double(in.intrinsics.ppv)<<"width"<<640<<"height"<<512<<"}"
      <<"setup_us"<<setup_us<<"ground_truth_file"<<in.truth.generic_u8string();
  std::ofstream json{directory/("metadata_"+title+".json")};json<<meta.releaseAndGetString();json.close();Require(!json.fail(),"Metadata write failed");
  cv::Mat overlay=initial.clone();std::vector<cv::Point> pixels;std::vector<bool> visible;
  for(const auto &v:in.vertices) {const Eigen::Vector3f p=in.poses.front()*v;visible.push_back(p.z()>.01f);pixels.emplace_back(int(in.intrinsics.fu*p.x()/p.z()+in.intrinsics.ppu),int(in.intrinsics.fv*p.y()/p.z()+in.intrinsics.ppv));}
  for(const auto &t:in.triangles)for(int j=0;j<3;++j) {int a=t[j],b=t[(j+1)%3];if(visible[a] && visible[b])cv::line(overlay,pixels[a],pixels[b],cv::Scalar(200,60,220),1,cv::LINE_AA);}
  fs::create_directories(in.output/"results/overlays");Require(cv::imwrite((in.output/"results/overlays/initial_gt_image_0.png").string(),overlay),"Projection write failed");
}
'''

OURS_PREFIX = r'''
// RBOT final driver: same final setters, no diagnostic collection in timing runs.
#include "rbot_evaluator.h"
#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
int main(int argc,char **argv) {
  try {
    const bool benchmark_mode=true;
    std::map<std::string,std::string> args;
    const std::set<std::string> keys{"--dataset","--cache","--output","--body","--sequence","--frames","--poses","--disk-model"};
    for(int i=1;i<argc;++i) {const std::string key=argv[i];if(!keys.count(key) || ++i>=argc || args.count(key))throw std::runtime_error("Invalid argument: "+key);args[key]=argv[i];}
    for(const auto &key:keys)if(!args.count(key))throw std::runtime_error("Missing "+key);
    const std::filesystem::path root=args.at("--dataset"),output=args.at("--output");
    const std::string object=args.at("--body"),sequence=args.at("--sequence");
    size_t used=0;const int frames=std::stoi(args.at("--frames"),&used);
    if(used!=args.at("--frames").size() || frames<2 || frames>1001)throw std::runtime_error("Frames must be 2..1001");
    RBOTEvaluator::ExternalSequence input;
    input.image_directory=root/object/"frames";input.model_file=root/object/(object+".obj");
    input.pose_file=args.at("--poses");input.intrinsics={650.048f,647.183f,323.828f,256.823f,640,512};
    input.image_count=frames;input.pose_row_count=1001;input.first_image_index=0;input.leading_zeros=4;input.image_prefix=sequence;
    input.geometry_unit_in_meter=.001f;input.pose_translation_scale=1.0f;
    if(!std::filesystem::is_regular_file(input.model_file) || !std::filesystem::is_regular_file(args.at("--disk-model")))throw std::runtime_error("Missing CAD/DISK");
    RBOTEvaluator evaluator{"rbot_comparison",root,args.at("--cache"),{object},{sequence},{false}};
'''


def generate(baseline, current, output):
    baseline, current, output = Path(baseline), Path(current), Path(output)
    source = baseline.read_text(encoding='utf-8-sig')
    prefix = source[:source.index('struct Input {')]
    engine = source[source.index('struct Times {'):source.index('void Save(')]
    assert engine.count('in.mesh,1.0f,true,false') == 3
    engine = engine.replace('in.mesh,1.0f,true,false', 'in.mesh,.001f,true,false')
    main = source[source.index('int main('):].replace('Baseline evaluation failed:', 'RBOT evaluation failed:')
    baseline_code = prefix + COMMON_INPUT + CAMERA + engine + SAVE + main
    ours = (current/'examples/evaluate_industrial_dataset.cpp').read_text(encoding='utf-8-sig')
    setters = ours[ours.index('    evaluator.set_external_sequence(input);'):ours.index('    std::cout << "Industrial final:')]
    tail = '''    std::cout << "RBOT final " << object << '/' << sequence << "; evaluating images 1.." << frames-1 << std::endl;
    if(!evaluator.SetUp() || !evaluator.Evaluate())return 1;
    return 0;
  } catch(const std::exception &e) {std::cerr<<"RBOT final failed: "<<e.what()<<std::endl;return 1;}
}
'''
    ours_code = OURS_PREFIX + setters + tail
    output.mkdir(parents=True, exist_ok=True)
    for name, code in [('rbot_baseline.cpp', baseline_code), ('rbot_ours.cpp', ours_code)]:
        (output/name).write_text(code, encoding='utf-8')
    return {'baseline_template_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
            'ours_setters_template_sha256': hashlib.sha256((current/'examples/evaluate_industrial_dataset.cpp').read_bytes()).hexdigest(),
            'generated_sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                 for name in ['rbot_baseline.cpp', 'rbot_ours.cpp']}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.baseline, a.current, a.output), indent=2))
