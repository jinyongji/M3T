// SPDX-License-Identifier: MIT
// Industrial RGB adapters for the supplied, unmodified DLR algorithm sources.
// Parameters/cycle ordering follow each algorithm's own RBOT evaluator.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(BENCH_RBGT)
#include <rbgt/body.h>
#include <rbgt/camera.h>
#include <rbgt/model.h>
#include <rbgt/region_modality.h>
#include <rbgt/renderer_geometry.h>
#include <rbgt/tracker.h>
namespace engine = rbgt;
constexpr const char *kMethod = "rbgt";
#elif defined(BENCH_SRT3D)
#include <srt3d/body.h>
#include <srt3d/camera.h>
#include <srt3d/model.h>
#include <srt3d/region_modality.h>
#include <srt3d/renderer_geometry.h>
#include <srt3d/tracker.h>
namespace engine = srt3d;
constexpr const char *kMethod = "srt3d";
#elif defined(BENCH_ICG)
#include <icg/body.h>
#include <icg/camera.h>
#include <icg/region_model.h>
#include <icg/region_modality.h>
#include <icg/optimizer.h>
#include <icg/tracker.h>
namespace engine = icg;
constexpr const char *kMethod = "icg_rgb";
#else
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/region_model.h>
#include <m3t/region_modality.h>
#include <m3t/texture_modality.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <m3t/tracker.h>
namespace engine = m3t;
constexpr const char *kMethod = "m3t_original_rgb";
#endif

namespace fs = std::filesystem;
using Pose = Eigen::Transform<float, 3, Eigen::Affine>;
using Clock = std::chrono::steady_clock;
double Micros(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now()-start).count();
}
void Require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
float LegacyModelDiameter(float enclosing_diameter) {
  // RBGT/SRT3D use f=(image_size-20)*sphere_radius/diameter.
  // A sphere of radius r at distance R subtends tan(asin(r/R)), not r/R.
  // Inflate only their rendering bound, keeping mesh coordinates and GT intact.
  constexpr double sphere_radius=.8, raster_margin=1.02;
  const double radius=.5*enclosing_diameter;
  Require(std::isfinite(radius) && radius>0 && radius<sphere_radius,
          "Legacy model radius must lie inside the 0.8m camera sphere");
  const double padded=2*radius/std::sqrt(1-radius*radius/(sphere_radius*sphere_radius))*raster_margin;
  Require(padded<2*sphere_radius, "Legacy model near clip would be non-positive; use a separately validated larger sphere");
  return static_cast<float>(padded);
}

template<class F> double Measured(F function) {
  const auto start=Clock::now();
  Require(function(), "Tracker step failed");
  return Micros(start);
}

struct Input {
  fs::path root, cache, output, images, mesh, truth;
  std::string scene, motion, condition, object, sequence;
  int frames=500;
  engine::Intrinsics intrinsics{};
  std::vector<Pose> poses;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<std::array<int,3>> triangles;
  float diameter=0,legacy_model_diameter=0;
};

Input ReadInput(int argc, char **argv) {
  std::map<std::string,std::string> args;
  const std::vector<std::string> required{"--dataset","--cache","--output","--scene","--motion","--condition","--object","--frames"};
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];
    Require(std::find(required.begin(),required.end(),key)!=required.end() && !args.count(key) && i+1<argc,"Invalid/duplicate argument: "+key);
    args[key]=argv[++i];
  }
  for(const auto &key:required) Require(args.count(key),"Missing "+key);
  Input in;
  in.root=args["--dataset"];in.cache=args["--cache"];in.output=args["--output"];
  in.scene=args["--scene"];in.motion=args["--motion"];in.condition=args["--condition"];in.object=args["--object"];
  for(const auto &part:{in.scene,in.motion,in.condition,in.object})
    Require(!part.empty() && part!="." && part!=".." && part.find_first_of("/\\")==std::string::npos,"Invalid path component");
  size_t parsed=0;in.frames=std::stoi(args["--frames"],&parsed);
  Require(parsed==args["--frames"].size() && in.frames>=2 && in.frames<=500,"Frames must be 2..500 including GT initialization");
  in.sequence=in.scene+"__"+in.motion+"__"+in.condition;
  in.images=in.root/"images"/in.scene/in.motion/in.condition;
  in.mesh=in.root/"objects"/in.object/(in.object+".obj");
  in.truth=in.root/"poses"/in.scene/in.motion/in.condition/("poses_"+in.object.substr(4)+".txt");
  cv::FileStorage camera{(in.root/"cameras/camera.json").string(),cv::FileStorage::READ};
  Require(camera.isOpened(),"Cannot open camera.json");
  const double expected[3][3]{{1125,0,449.5},{0,1125,299.5},{0,0,1}};
  const auto k=camera["K"];
  Require(k.isSeq() && k.size()==3,"Invalid K");
  for(int r=0;r<3;++r) {
    Require(k[r].isSeq() && k[r].size()==3,"Invalid K row");
    for(int c=0;c<3;++c) Require(std::isfinite(double(k[r][c])) && std::abs(double(k[r][c])-expected[r][c])<1e-6,"Expected the 900x600 calibration");
  }
  Require(int(camera["width"])==900 && int(camera["height"])==600 && int(camera["frames_per_sequence"])==500 &&
          std::string(camera["translation_unit"])=="meter" && std::string(camera["pose_direction"])=="p_camera = R @ p_object + t","Unsupported dataset convention");
  for(const auto &v:camera["distortion_coefficients"]) Require(double(v)==0,"Distortion is not supported");
  in.intrinsics={1125,1125,449.5f,299.5f,900,600};
  std::ifstream truth{in.truth};Require(bool(truth),"Cannot open pose input");
  std::string line;
  while(std::getline(truth,line)) {
    const auto first=line.find_first_not_of(" \t\r");
    if(first==std::string::npos || line[first]=='#') continue;
    std::istringstream values{line};values.imbue(std::locale::classic());Pose pose=Pose::Identity();
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)Require(bool(values>>pose.matrix()(r,c)),"Invalid pose rotation");
    for(int r=0;r<3;++r)Require(bool(values>>pose.matrix()(r,3)),"Invalid pose translation");
    std::string extra;Require(!(values>>extra),"Extra pose fields");
    Require(pose.matrix().allFinite() && std::abs(pose.linear().determinant()-1)<1e-3 &&
      (pose.linear().transpose()*pose.linear()-Eigen::Matrix3f::Identity()).cwiseAbs().maxCoeff()<1e-3,"Invalid GT pose");
    in.poses.push_back(pose);
  }
  Require(in.poses.size()==500,"GT must contain all 500 rows");
  std::ifstream mesh{in.mesh};Require(bool(mesh),"Cannot open OBJ");
  while(std::getline(mesh,line)) {
    std::istringstream row{line};std::string kind;row>>kind;
    if(kind=="v") {
      Eigen::Vector3f p;Require(bool(row>>p.x()>>p.y()>>p.z()) && p.allFinite(),"Invalid OBJ vertex");
      in.vertices.push_back(p);in.diameter=std::max(in.diameter,2*p.norm());
    } else if(kind=="f") {
      std::vector<int> face;std::string token;
      while(row>>token) {int index=std::stoi(token.substr(0,token.find('/')));index=index>0?index-1:int(in.vertices.size())+index;face.push_back(index);}
      for(size_t i=1;i+1<face.size();++i)in.triangles.push_back({face[0],face[i],face[i+1]});
    }
  }
  Require(in.diameter>0 && in.diameter<1.6f,"CAD model does not fit the fixed 0.8m model sphere");
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  in.legacy_model_diameter=LegacyModelDiameter(in.diameter);
  std::cout<<"Legacy template bounds: geometric diameter="<<in.diameter
           <<" m; perspective-safe render diameter="<<in.legacy_model_diameter<<" m"<<std::endl;
#endif
  for(const auto &t:in.triangles)for(int i:t)Require(i>=0 && i<int(in.vertices.size()),"Invalid OBJ index");
  fs::create_directories(in.cache);fs::create_directories(in.output/"results/poses");
  return in;
}

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
    // SRT3D's tracker calls SetUp on all components again. Do not consume a
    // second image: every method must initialize on exactly image 000001.
    if (index_ > 1) return true;
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
    std::ostringstream filename;filename<<std::setw(6)<<std::setfill('0')<<index_<<".png";
    const auto begin=Clock::now();image_=cv::imread((input_.images/filename.str()).string(),cv::IMREAD_UNCHANGED);decode_us=Micros(begin);
    Require(!image_.empty() && image_.rows==600 && image_.cols==900 && image_.type()==CV_16UC3,"Expected a complete 900x600 uint16 RGB frame: "+filename.str());
    const auto convert=Clock::now();image_.convertTo(image_,CV_8UC3,1.0/257.0);preprocess_us=Micros(convert);++index_;
    return true;
  }
  const Input &input_;int index_=1;
};

struct Times {double before=0,correspondences=0,gradient=0,optimization=0,results=0,wall=0;
  double Native()const{return before+correspondences+gradient+optimization+results;}};

class Engine {
 public:
  explicit Engine(const Input &in):camera(std::make_shared<IndustrialCamera>(in)) {
    Require(camera->SetUp(),"Camera setup failed");
#if defined(BENCH_RBGT)
    body=std::make_shared<engine::Body>(in.object,in.mesh,1.0f,true,false,in.legacy_model_diameter,Pose::Identity());
    body->set_occlusion_mask_id(7);
    geometry=std::make_shared<engine::RendererGeometry>();Require(geometry->Init(),"GL context failed");Require(geometry->AddBody(body),"Add body failed");
    model=std::make_shared<engine::Model>(in.object+"_model");model->set_use_random_seed(false);
    if(!model->LoadModel(in.cache,in.object)) {
      Require(model->GenerateModel(*body,.8f,4,200),"RBGT model generation failed");
      Require(model->SaveModel(in.cache,in.object),"RBGT model cache write failed");
    }
    region=std::make_shared<engine::RegionModality>();Require(region->Init("region",body,model,camera),"RBGT region init failed");
    tracker=std::make_shared<engine::Tracker>();tracker->AddRegionModality(region);tracker->SetUpObjects();
#elif defined(BENCH_SRT3D)
    body=std::make_shared<engine::Body>(in.object,in.mesh,1.0f,true,false,in.legacy_model_diameter,Pose::Identity(),7);
    geometry=std::make_shared<engine::RendererGeometry>("geometry");Require(geometry->SetUp(),"GL context failed");Require(geometry->AddBody(body),"Add body failed");
    model=std::make_shared<engine::Model>("model",body,in.cache,in.object+"_model.bin",.8f,4,200,false,2000);
    Require(model->SetUp(),"SRT3D model setup failed");
    region=std::make_shared<engine::RegionModality>("region",body,model,camera);Require(region->SetUp(),"SRT3D region setup failed");
    tracker=std::make_shared<engine::Tracker>("tracker");tracker->AddRegionModality(region);Require(tracker->SetUpTracker(),"SRT3D tracker setup failed");
#else
    body=std::make_shared<engine::Body>(in.object,in.mesh,1.0f,true,false,Pose::Identity());Require(body->SetUp(),"Body setup failed");
    geometry=std::make_shared<engine::RendererGeometry>("geometry");Require(geometry->SetUp(),"GL context failed");Require(geometry->AddBody(body),"Add body failed");
    model=std::make_shared<engine::RegionModel>("model",body,in.cache/(in.object+"_model.bin"),.8f,4,200,.01f,.002f,false,2000);
    Require(model->SetUp(),"Region model setup failed");region=std::make_shared<engine::RegionModality>("region",body,camera,model);
    ConfigureRegion();Require(region->SetUp(),"Region setup failed");tracker=std::make_shared<engine::Tracker>("tracker");
#if defined(BENCH_ICG)
    optimizer=std::make_shared<engine::Optimizer>("optimizer");optimizer->AddModality(region);
    optimizer->set_tikhonov_parameter_rotation(1000);optimizer->set_tikhonov_parameter_translation(100000);
#else
    silhouette=std::make_shared<engine::FocusedSilhouetteRenderer>("silhouette",geometry,camera);silhouette->AddReferencedBody(body);Require(silhouette->SetUp(),"Silhouette setup failed");
    texture=std::make_shared<engine::TextureModality>("texture",body,camera,silhouette);ConfigureTexture();Require(texture->SetUp(),"Texture setup failed");
    link=std::make_shared<engine::Link>("link",body);link->AddModality(region);link->AddModality(texture);Require(link->SetUp(),"Link setup failed");
    optimizer=std::make_shared<engine::Optimizer>("optimizer",link);optimizer->set_tikhonov_parameter_rotation(1000);optimizer->set_tikhonov_parameter_translation(30000);
#endif
    Require(optimizer->SetUp(),"Optimizer setup failed");tracker->AddOptimizer(optimizer);Require(tracker->SetUp(false),"Tracker setup failed");
#endif
    tracker->set_n_corr_iterations(7);tracker->set_n_update_iterations(2);
    Reset(in.poses.front());
  }
  void Reset(const Pose &pose) {
    body->set_body2world_pose(pose);
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
    Require(region->StartModality(),"Region restart failed");
#else
#if !defined(BENCH_ICG)
    for(const auto &r:tracker->start_modality_renderer_ptrs())Require(r->StartRendering(),"Start rendering failed");
#endif
    Require(region->StartModality(0,0),"Region restart failed");
#if !defined(BENCH_ICG)
    Require(texture->StartModality(0,0),"Texture restart failed");
#endif
#endif
  }
  Times Step(int iteration) {
    Times t;const auto start=Clock::now();
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
    t.before=Measured([&]{return tracker->CalculateBeforeCameraUpdate();});Require(tracker->UpdateCameras(),"Image update failed");
    for(int c=0;c<tracker->n_corr_iterations();++c) {
#if defined(BENCH_RBGT)
      t.correspondences+=Measured([&]{return tracker->StartOcclusionMaskRendering();});
#else
      t.correspondences+=Measured([&]{return tracker->StartOcclusionRendering();});
#endif
      t.correspondences+=Measured([&]{return tracker->CalculateCorrespondences(c);});
      for(int u=0;u<tracker->n_update_iterations();++u) {
#if defined(BENCH_RBGT)
        t.optimization+=Measured([&]{return tracker->CalculatePoseUpdate();});
#else
        t.optimization+=Measured([&]{return tracker->CalculatePoseUpdate(c,u);});
#endif
      }
    }
#else
#if defined(BENCH_ICG)
    Require(tracker->UpdateCameras(false),"Image update failed");
#else
    Require(tracker->UpdateCameras(iteration),"Image update failed");
#endif
    for(int c=0;c<tracker->n_corr_iterations();++c) {
      t.correspondences+=Measured([&]{return tracker->CalculateCorrespondences(iteration,c);});
      for(int u=0;u<tracker->n_update_iterations();++u) {
        t.gradient+=Measured([&]{return tracker->CalculateGradientAndHessian(iteration,c,u);});
        t.optimization+=Measured([&]{return tracker->CalculateOptimization(iteration,c,u);});
      }
    }
    t.results=Measured([&]{return tracker->CalculateResults(iteration);});
#endif
    t.wall=Micros(start);return t;
  }
  Pose Estimated()const{return body->body2world_pose();}
  std::shared_ptr<IndustrialCamera> camera;
 private:
#if !defined(BENCH_RBGT) && !defined(BENCH_SRT3D)
  void ConfigureRegion() {
#if defined(BENCH_ICG)
    region->set_n_lines(200);region->set_min_continuous_distance(6);
    region->set_standard_deviations({15,5,3.5f,1.5f});region->set_unconsidered_line_length(1);region->set_max_considered_line_length(18);
#else
    region->set_n_lines_max(200);region->set_use_adaptive_coverage(false);region->set_min_continuous_distance(3);
    region->set_standard_deviations({20,7,3,1.5f});region->set_unconsidered_line_length(.5f);region->set_max_considered_line_length(20);
    region->set_n_unoccluded_iterations(0);
#endif
    region->set_function_length(8);region->set_distribution_length(12);region->set_function_amplitude(.36f);region->set_function_slope(0);
    region->set_learning_rate(1.3f);region->set_scales({5,2,2,1});region->set_n_histogram_bins(32);region->set_learning_rate_f(.2f);region->set_learning_rate_b(.2f);
    region->set_modeled_depth_offset_radius(0);region->set_modeled_occlusion_radius(0);region->set_modeled_occlusion_threshold(.03f);
  }
#endif
#if defined(BENCH_M3T)
  void ConfigureTexture() {
    texture->set_descriptor_type(engine::TextureModality::DescriptorType::ORB);texture->set_focused_image_size(200);
    texture->set_descriptor_distance_threshold(.7f);texture->set_tukey_norm_constant(20);texture->set_standard_deviations({5,1});
    texture->set_max_keyframe_rotation_difference(10*engine::kPi/180);texture->set_max_keyframe_age(1000);texture->set_n_keyframes(1);
    texture->set_orb_n_features(300);texture->set_orb_scale_factor(1.2f);texture->set_orb_n_levels(3);
    texture->set_modeled_occlusion_radius(0);texture->set_modeled_occlusion_threshold(.03f);
  }
#endif
  std::shared_ptr<engine::Body> body;
  std::shared_ptr<engine::RendererGeometry> geometry;
  std::shared_ptr<engine::Tracker> tracker;
  std::shared_ptr<engine::RegionModality> region;
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  std::shared_ptr<engine::Model> model;
#else
  std::shared_ptr<engine::RegionModel> model;
  std::shared_ptr<engine::Optimizer> optimizer;
#if defined(BENCH_M3T)
  std::shared_ptr<engine::Link> link;
  std::shared_ptr<engine::TextureModality> texture;
  std::shared_ptr<engine::FocusedSilhouetteRenderer> silhouette;
#endif
#endif
};

struct Row {Pose estimate,truth;Times times;double translation,rotation,metric_us,reset_us,decode_us,preprocess_us;bool success;};
void EvaluatePose(Row *row) {
  const auto start=Clock::now();const auto &e=row->estimate;const auto &g=row->truth;
  const float det=e.linear().determinant();
  if(!e.matrix().allFinite() || !std::isfinite(det) || det<=1e-8f) {
    row->translation=row->rotation=std::numeric_limits<double>::infinity();row->success=false;
  } else {
    const float translation=(e.translation()-g.translation()).norm();
    const float cosine=((e.rotation().transpose()*g.rotation()).trace()-1.0f)/2.0f;
    const float rotation=std::acos(std::clamp(cosine,-1.0f,1.0f));
    row->translation=translation;row->rotation=rotation;
    row->success=std::isfinite(translation) && std::isfinite(rotation) && translation<=.05f && rotation<=5*engine::kPi/180;
  }
  row->metric_us=Micros(start);
}

void Save(const Input &in,const std::vector<Row> &rows,double setup_us,const cv::Mat &initial) {
  const auto directory=in.output/"results/poses";const auto title=in.sequence+"_"+in.object;
  std::ofstream out{directory/("poses_"+title+".csv")};out.imbue(std::locale::classic());out<<std::setprecision(17);
  out<<"frame,image_index,image_file,segment_id,estimate_source,pose_finite,reset_primary_after_frame,reset_secondary_after_frame,tracking_success,translation_error_m,rotation_error_rad,cycle_us";
  for(const auto *prefix:{"est","gt"}) {for(int r=0;r<3;++r)for(int c=0;c<3;++c)out<<','<<prefix<<"_r"<<r<<c;out<<','<<prefix<<"_tx_m,"<<prefix<<"_ty_m,"<<prefix<<"_tz_m";}
  out<<",decode_us,preprocessing_us,tracking_wall_us,compute_wall_us,metric_us,reset_us,frame_wall_us,before_camera_us,correspondences_us,gradient_hessian_us,optimization_us,results_us\n";
  int segment=0;for(size_t i=0;i<rows.size();++i) {
    const auto &r=rows[i];out<<i<<','<<i+2<<','<<std::setw(6)<<std::setfill('0')<<i+2<<".png,"<<segment<<",single,"<<int(r.estimate.matrix().allFinite())<<','<<int(!r.success)<<",0,"<<int(r.success)<<','<<r.translation<<','<<r.rotation<<','<<r.times.Native();
    for(const auto *p:{&r.estimate,&r.truth}) {for(int y=0;y<3;++y)for(int x=0;x<3;++x)out<<','<<p->matrix()(y,x);for(int y=0;y<3;++y)out<<','<<p->translation()(y);}
    out<<','<<r.decode_us<<','<<r.preprocess_us<<','<<r.times.wall<<','<<std::max(0.0,r.times.wall-r.decode_us)<<','<<r.metric_us<<','<<r.reset_us<<','<<r.times.wall+r.metric_us+r.reset_us
       <<','<<r.times.before<<','<<r.times.correspondences<<','<<r.times.gradient<<','<<r.times.optimization<<','<<r.times.results<<'\n';
    segment+=!r.success;
  }
  out.close();Require(!out.fail(),"Cannot write pose CSV");
  cv::FileStorage meta{"",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON};
  meta<<"schema_version"<<1<<"dataset_kind"<<"industrial"<<"method"<<kMethod<<"object"<<in.object<<"sequence"<<in.sequence
      <<"translation_unit"<<"m"<<"model_unit_in_meter"<<1.0<<"ground_truth_translation_scale"<<1.0
      <<"pose_convention"<<"p_camera = R * p_body + t"<<"initial_image_index"<<1<<"first_image_index"<<2<<"last_image_index"<<in.frames<<"frame_count"<<int(rows.size())
      <<"reset_protocol"<<"industrial_selected_output_reset"<<"estimate_capture"<<"final_selected_output_before_gt_reset"
      <<"input_conversion"<<"uint16_to_uint8_fixed_1_over_257"<<"translation_threshold_m"<<.05<<"rotation_threshold_rad"<<double(5*engine::kPi/180)
      <<"intrinsics"<<"{"<<"fx"<<1125<<"fy"<<1125<<"cx"<<449.5<<"cy"<<299.5<<"width"<<900<<"height"<<600<<"}"
      <<"initial_gt_pose_camera_row_major"<<"[";
  for(int y=0;y<4;++y)for(int x=0;x<4;++x)meta<<double(in.poses[0].matrix()(y,x));
  meta<<"]"<<"setup_us"<<setup_us<<"image_directory"<<in.images.generic_u8string()<<"ground_truth_file"<<in.truth.generic_u8string()
      <<"parameter_source"<<"supplied algorithm RGB RBOT example/defaults; 7x2 iterations; 200 model points; sphere 0.8m"
      <<"overlay_type"<<"GT projected CAD wireframe; visualization only";
  meta<<"geometry_enclosing_diameter_m"<<double(in.diameter);
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  meta<<"legacy_model_render_diameter_m"<<double(in.legacy_model_diameter)
      <<"legacy_model_bounds_policy"<<"perspective sphere bound at 0.8m plus 2 percent raster margin; mesh/GT unchanged";
#endif
  std::ofstream json{directory/("metadata_"+title+".json")};json<<meta.releaseAndGetString();json.close();Require(!json.fail(),"Cannot write metadata");
  cv::Mat overlay=initial.clone();std::vector<cv::Point> pixels;std::vector<bool> visible;
  for(const auto &v:in.vertices) {const Eigen::Vector3f p=in.poses.front()*v;visible.push_back(p.z()>.01f);pixels.emplace_back(int(1125*p.x()/p.z()+449.5f),int(1125*p.y()/p.z()+299.5f));}
  for(const auto &triangle:in.triangles)for(int j=0;j<3;++j) {const int a=triangle[j],b=triangle[(j+1)%3];if(visible[a] && visible[b])cv::line(overlay,pixels[a],pixels[b],cv::Scalar(200,60,220),1,cv::LINE_AA);}
  fs::create_directories(in.output/"results/overlays");Require(cv::imwrite((in.output/"results/overlays/initial_gt_image_1.png").string(),overlay),"Cannot write initial projection");
}

int main(int argc,char **argv) {
  try {
    cv::setNumThreads(1);const auto setup_start=Clock::now();const Input input=ReadInput(argc,argv);Engine tracker{input};
    const double setup_us=Micros(setup_start);const cv::Mat initial=tracker.camera->image().clone();std::vector<Row> rows;
    for(int i=0;i<input.frames-1;++i) {
      const auto frame_begin=Clock::now();
      Row row;row.times=tracker.Step(i);row.estimate=tracker.Estimated();row.truth=input.poses[i+1];EvaluatePose(&row);row.reset_us=0;
      row.decode_us=tracker.camera->decode_us;row.preprocess_us=tracker.camera->preprocess_us;
      if(!row.success) {const auto start=Clock::now();tracker.Reset(row.truth);row.reset_us=Micros(start);}
      row.times.wall=std::max(0.0,Micros(frame_begin)-row.metric_us-row.reset_us);
      rows.push_back(row);
    }
    Save(input,rows,setup_us,initial);std::cout<<"Complete "<<kMethod<<" frames="<<rows.size()<<std::endl;
  } catch(const std::exception &e) {std::cerr<<"Baseline evaluation failed: "<<e.what()<<std::endl;return 1;}
}
