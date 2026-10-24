// Generated prefix and frozen engine body are injected by tracking_build.py.
struct View {int image,dx,dy;double visibility;Pose gt;};
struct Input {
  fs::path root,cache,output,images,mesh,disk_model;std::string object,id;
  int frames=72;engine::Intrinsics intrinsics{};double diameter=0;float legacy_model_diameter=0;
  std::vector<View> views;std::vector<Pose> poses;
  std::vector<Eigen::Vector3d> evaluation;std::vector<Eigen::Affine3d> symmetries;
};
Input ReadInput(int argc,char **argv) {
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2) {Require(i+1<argc,"Missing argument value");Require(!args.count(argv[i]),"Duplicate argument");args[argv[i]]=argv[i+1];}
  for(const auto *n:{"--dataset","--prepared","--id","--output","--cache","--frames","--disk-model"})Require(args.count(n),std::string("Missing ")+n);
  Require(args.size()==7,"Unknown argument");Input in;in.root=args.at("--dataset");in.cache=args.at("--cache");in.output=args.at("--output");in.disk_model=args.at("--disk-model");in.id=args.at("--id");
  Require(in.id.find_first_of("/\\.")==std::string::npos,"Invalid configuration id");
  const fs::path prepared=args.at("--prepared");std::ifstream camera{prepared/"camera.txt"};
  double fx,fy,cx,cy;int width,height;Require(bool(camera>>fx>>fy>>cx>>cy>>width>>height),"Invalid camera.txt");
  in.intrinsics={float(fx),float(fy),float(cx),float(cy),width,height};
  std::ifstream frames{prepared/"configurations"/in.id/"frames.txt"};int total;
  Require(bool(frames>>in.object>>in.diameter>>total),"Invalid frame header");
  size_t used=0;in.frames=std::stoi(args.at("--frames"),&used);Require(used==args.at("--frames").size()&&in.frames>=2&&in.frames<=total,"Invalid frame limit");
  for(int i=0;i<total;++i) {View v;v.gt=Pose::Identity();Require(bool(frames>>v.image>>v.dx>>v.dy>>v.visibility),"Missing view");
    for(int y=0;y<3;++y)for(int x=0;x<3;++x)Require(bool(frames>>v.gt.matrix()(y,x)),"Invalid GT rotation");
    for(int y=0;y<3;++y)Require(bool(frames>>v.gt.translation()(y)),"Invalid GT translation");
    Require(v.dx>=0&&v.dy>=0&&v.dx+720<=width&&v.dy+540<=height&&v.gt.matrix().allFinite(),"Invalid padding/pose");in.views.push_back(v);in.poses.push_back(v.gt);
  }
  const auto p=in.id.find("__");const int scene=std::stoi(in.id.substr(6,p-6));std::ostringstream folder;folder<<std::setw(6)<<std::setfill('0')<<scene;
  in.images=in.root/"test_primesense"/folder.str()/"rgb";const auto model=prepared/"models"/in.object;in.mesh=model/"model.obj";
  std::ifstream xyz{model/"evaluation.xyz"};double x,y,z;while(xyz>>x>>y>>z)in.evaluation.emplace_back(x,y,z);
  Require(xyz.eof()&&!in.evaluation.empty(),"Invalid evaluation vertices");
  std::ifstream syms{model/"symmetries.txt"};std::string line;
  while(std::getline(syms,line)) {std::istringstream s{line};Eigen::Affine3d p=Eigen::Affine3d::Identity();
    for(int y=0;y<3;++y)for(int x=0;x<3;++x)Require(bool(s>>p.matrix()(y,x)),"Invalid symmetry");
    for(int y=0;y<3;++y)Require(bool(s>>p.translation()(y)),"Invalid symmetry offset");in.symmetries.push_back(p);}
  Require(!in.symmetries.empty(),"Missing symmetry set");
  std::ifstream mesh{in.mesh};double radius=0;
  while(std::getline(mesh,line)) {std::istringstream s{line};std::string kind;s>>kind;if(kind=="v"){Require(bool(s>>x>>y>>z),"Invalid OBJ vertex");radius=std::max(radius,std::sqrt(x*x+y*y+z*z));}}
  Require(radius>0&&radius<.8,"Invalid model sphere");
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  in.legacy_model_diameter=LegacyModelDiameter(float(2*radius));
#endif
  fs::create_directories(in.cache);fs::create_directories(in.output);return in;
}

#if defined(BENCH_OURS)
using CameraBase=engine::LoaderColorCamera;
#elif defined(BENCH_RBGT) || defined(BENCH_SRT3D)
using CameraBase=engine::Camera;
#else
using CameraBase=engine::ColorCamera;
#endif
class IndustrialCamera:public CameraBase {
 public:
  explicit IndustrialCamera(const Input &in)
#if defined(BENCH_OURS)
    :CameraBase("camera",in.images,in.intrinsics),input_(in) {set_use_noise_adaptive_filtering(true);set_use_gaussian_noise_filter(true);}
#elif defined(BENCH_RBGT)
    :CameraBase(),input_(in) {name_="camera";intrinsics_=in.intrinsics;}
#else
    :CameraBase("camera"),input_(in) {intrinsics_=in.intrinsics;}
#endif
  bool SetUp()
#if !defined(BENCH_RBGT)
    override
#endif
  {if(index_>0)return true;
#if defined(BENCH_RBGT)
    initialized_=true;
#else
    set_up_=true;
#endif
    return Load();}
#if defined(BENCH_RBGT) || defined(BENCH_SRT3D)
  bool UpdateImage()override{return Load();}
#else
  bool UpdateImage(bool)override{return Load();}
#endif
  double decode_us=0,preprocess_us=0;
 private:
  bool Load() {
    Require(index_<input_.frames,"Camera consumed extra image");const auto &view=input_.views[index_];std::ostringstream name;name<<std::setw(6)<<std::setfill('0')<<view.image<<".png";
    const auto start=Clock::now();cv::Mat raw=cv::imread((input_.images/name.str()).string(),cv::IMREAD_UNCHANGED);decode_us=Micros(start);
    Require(!raw.empty()&&raw.type()==CV_8UC3&&raw.cols==720&&raw.rows==540,"Expected native RGB PNG: "+name.str());
    const auto process=Clock::now();cv::Mat padded;
    cv::copyMakeBorder(raw,padded,view.dy,input_.intrinsics.height-540-view.dy,view.dx,input_.intrinsics.width-720-view.dx,cv::BORDER_CONSTANT,cv::Scalar(127,127,127));
#if defined(BENCH_OURS)
    SetAdapterFrame(padded); // Original LoaderColorCamera noise code, original raw-ORB accessor.
#else
    image_=padded;
#endif
    preprocess_us=Micros(process);++index_;return true;
  }
  const Input &input_;int index_=0;
};
struct Times {double before=0,correspondences=0,gradient=0,optimization=0,results=0,wall=0;
  double Native()const{return before+correspondences+gradient+optimization+results;}};

// @@ENGINE@@

struct Errors {double mssd=std::numeric_limits<double>::infinity(),mspd=mssd;bool finite=false;};
Errors Evaluate(const Input &in,const Pose &estimate,const Pose &truth) {
  Errors result;const auto p=estimate.matrix().cast<double>();
  if(!p.allFinite()||p.topLeftCorner<3,3>().determinant()<=1e-8)return result;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(p.topLeftCorner<3,3>(),Eigen::ComputeFullU|Eigen::ComputeFullV);
  const Eigen::Matrix3d rotation=svd.matrixU()*svd.matrixV().transpose();if(rotation.determinant()<=0)return result;
  std::vector<Eigen::Vector3d> points;std::vector<Eigen::Vector2d> pixels;points.reserve(in.evaluation.size());pixels.reserve(in.evaluation.size());
  const auto project=[&](const Eigen::Vector3d &v){return Eigen::Vector2d(in.intrinsics.fu*v.x()/v.z()+in.intrinsics.ppu,in.intrinsics.fv*v.y()/v.z()+in.intrinsics.ppv);};
  for(const auto &v:in.evaluation) {Eigen::Vector3d q=rotation*v+p.topRightCorner<3,1>();if(q.z()<=0)return result;points.push_back(q);pixels.push_back(project(q));}
  double best3=std::numeric_limits<double>::infinity(),best2=best3;
  const Eigen::Affine3d gt=truth.cast<double>();
  for(const auto &sym:in.symmetries) {
    const Eigen::Affine3d target=gt*sym;double max3=0,max2=0;
    for(size_t j=0;j<in.evaluation.size();++j) {
      const Eigen::Vector3d q=target*in.evaluation[j];if(q.z()<=0)return result;
      if(max3<best3)max3=std::max(max3,(q-points[j]).squaredNorm());
      if(max2<best2)max2=std::max(max2,(project(q)-pixels[j]).squaredNorm());
      if(max3>=best3&&max2>=best2)break;
    }
    best3=std::min(best3,max3);best2=std::min(best2,max2);
  }
  result.mssd=std::sqrt(best3);result.mspd=std::sqrt(best2);result.finite=true;return result;
}
void WritePose(std::ostream &s,const Pose &p) {for(int y=0;y<3;++y)for(int x=0;x<3;++x)s<<','<<p.matrix()(y,x);for(int y=0;y<3;++y)s<<','<<p.translation()(y);}
int main(int argc,char **argv) {
  try {
    cv::setNumThreads(1);cv::setRNGSeed(0);const auto start=Clock::now();const Input input=ReadInput(argc,argv);Engine tracker{input};const double setup_us=Micros(start);
    // Save a setup-only GT wireframe on the same normalized image.
    cv::Mat overlay=tracker.camera->image().clone();std::ifstream mesh{input.mesh};std::string line;std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<int,3>> faces;
    while(std::getline(mesh,line)){std::istringstream s{line};std::string kind;s>>kind;if(kind=="v"){float x,y,z;s>>x>>y>>z;vertices.emplace_back(x,y,z);}else if(kind=="f"){std::array<int,3> f;for(int j=0;j<3;++j){std::string token;Require(bool(s>>token),"Invalid OBJ overlay face");f[j]=std::stoi(token);}faces.push_back(f);}}
    for(const auto &f:faces)for(int j=0;j<3;++j){const Eigen::Vector3f a=input.poses[0]*vertices.at(f[j]-1),b=input.poses[0]*vertices.at(f[(j+1)%3]-1);if(a.z()>.01&&b.z()>.01)cv::line(overlay,cv::Point(int(input.intrinsics.fu*a.x()/a.z()+input.intrinsics.ppu),int(input.intrinsics.fv*a.y()/a.z()+input.intrinsics.ppv)),cv::Point(int(input.intrinsics.fu*b.x()/b.z()+input.intrinsics.ppu),int(input.intrinsics.fv*b.y()/b.z()+input.intrinsics.ppv)),cv::Scalar(200,60,220),1,cv::LINE_AA);}
    Require(cv::imwrite((input.output/"initial_gt.png").string(),overlay),"Cannot write overlay");
    std::ofstream out{input.output/"poses.csv"};out.imbue(std::locale::classic());out<<std::setprecision(17);
    out<<"frame,image_id,visibility,segment_id,pose_finite,success,reset_after,mssd_m,mspd_px";
    for(const auto *prefix:{"est","gt"}){for(int r=0;r<3;++r)for(int c=0;c<3;++c)out<<','<<prefix<<"_r"<<r<<c;for(const auto *axis:{"x","y","z"})out<<','<<prefix<<"_t"<<axis;}
    out<<",decode_us,preprocessing_us,tracking_wall_us,compute_wall_us,metric_us,reset_us,frame_wall_us,cycle_us,correspondences_us,gradient_hessian_us,optimization_us,results_us\n";
    int segment=0,successes=0;
    for(int i=1;i<input.frames;++i) {
      const auto frame_start=Clock::now();Times times=tracker.Step(i-1);const Pose estimate=tracker.Estimated(),gt=input.poses[i];
      const auto metric_start=Clock::now();const auto errors=Evaluate(input,estimate,gt);const bool success=errors.finite&&errors.mssd<=.1*input.diameter;const double metric_us=Micros(metric_start);
      double reset_us=0;
      if(!success){const auto reset_start=Clock::now();tracker.Reset(gt);reset_us=Micros(reset_start);}else{
#if defined(BENCH_OURS)
        tracker.Accept();
#endif
      }
      const double wall=std::max(0.,Micros(frame_start)-metric_us-reset_us),decode=tracker.camera->decode_us;
      Require(wall>=decode,"Decode exceeds total tracking time");out<<i-1<<','<<input.views[i].image<<','<<input.views[i].visibility<<','<<segment<<','<<errors.finite<<','<<success<<','<<!success<<','<<errors.mssd<<','<<errors.mspd;
      WritePose(out,estimate);WritePose(out,gt);out<<','<<decode<<','<<tracker.camera->preprocess_us<<','<<wall<<','<<wall-decode<<','<<metric_us<<','<<reset_us<<','<<wall+metric_us+reset_us<<','<<times.Native()<<','<<times.correspondences<<','<<times.gradient<<','<<times.optimization<<','<<times.results<<'\n';
      segment+=!success;successes+=success;
    }
    out.close();Require(!out.fail(),"CSV write failed");
    cv::FileStorage meta{(input.output/"metadata.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON};
    meta<<"method"<<kMethod<<"config_id"<<input.id<<"setup_us"<<setup_us<<"frame_count"<<input.frames-1<<"initial_image_id"<<input.views[0].image<<"mssd_reset_fraction"<<.1<<"diameter_m"<<input.diameter<<"symmetry_count"<<int(input.symmetries.size())<<"evaluation_vertices"<<int(input.evaluation.size())<<"estimate_capture"<<"before_gt_reset"<<"cad_texture_available"<<0<<"cad_disk_executed"<<0<<"input_normalization"<<"online_integer_shift_padding_no_interpolation";
#if defined(BENCH_OURS)
    meta<<"cad_disk_configured"<<1;
#else
    meta<<"cad_disk_configured"<<0;
#endif
    meta<<"intrinsics"<<"["<<input.intrinsics.fu<<input.intrinsics.fv<<input.intrinsics.ppu<<input.intrinsics.ppv<<input.intrinsics.width<<input.intrinsics.height<<"]";
    std::cout<<"Complete "<<kMethod<<" frames="<<input.frames-1<<" success="<<100.*successes/(input.frames-1)<<"%; CAD texture absent"<<std::endl;
  }catch(const std::exception &e){std::cerr<<"T-LESS evaluation failed: "<<e.what()<<std::endl;return 1;}
}
