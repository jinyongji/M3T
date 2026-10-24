// Separate adapter: frozen tracker sources, independent runtime ablation switches.
#include "rbot_evaluator.h"
#include <m3t/textured_renderer.h>
#include <map>
#include <set>
#include <stdexcept>

int main(int argc,char **argv) {
  try {
    std::map<std::string,std::string> args;
    const std::set<std::string> keys{"--input","--cache","--output","--disk-model","--flags","--variant"};
    for(int i=1;i<argc;i+=2) {
      if(i+1>=argc || !keys.count(argv[i]) || args.count(argv[i]))throw std::runtime_error("Invalid adapter arguments");
      args[argv[i]]=argv[i+1];
    }
    if(args.size()!=keys.size())throw std::runtime_error("Missing adapter arguments");
    const auto flags=args.at("--flags");
    if(flags.size()!=5 || flags.find_first_not_of("01")!=std::string::npos)throw std::runtime_error("Expected five D/F/P/T/N bits");
    const bool cad=flags[0]=='1',fusion=flags[1]=='1',prediction=flags[2]=='1',prior=flags[3]=='1',noise=flags[4]=='1';
    const bool benchmark_mode=true;
    cv::setNumThreads(1);
    cv::FileStorage data{args.at("--input"),cv::FileStorage::READ};
    if(!data.isOpened())throw std::runtime_error("Cannot read prepared input");
    const std::string object=data["object"],sequence=data["sequence"];
    const std::filesystem::path root=std::string(data["dataset"]),output=args.at("--output");
    RBOTEvaluator::ExternalSequence input;
    input.image_directory=std::string(data["images"]);input.model_file=std::string(data["mesh"]);input.pose_file=std::string(data["poses"]);
    const auto k=data["intrinsics"];
    input.intrinsics={float(k["fx"]),float(k["fy"]),float(k["cx"]),float(k["cy"]),int(k["width"]),int(k["height"])};
    input.image_count=int(data["frames"]);input.pose_row_count=int(data["pose_rows"]);
    input.first_image_index=int(data["first_image"]);input.leading_zeros=int(data["leading_zeros"]);input.image_prefix=std::string(data["prefix"]);
    input.geometry_unit_in_meter=float(data["geometry_scale"]);input.pose_translation_scale=1.0f;
    if(!std::filesystem::is_regular_file(input.model_file))throw std::runtime_error("Missing CAD");
    if(cad && !std::filesystem::is_regular_file(args.at("--disk-model")))throw std::runtime_error("Missing DISK model");
    RBOTEvaluator evaluator{"ablation",root,args.at("--cache"),{object},{sequence},{false}};
    // @@FROZEN_SETTERS@@
    evaluator.set_use_ablation_prediction_smoothing(true);
    evaluator.DisableDiagnostics();
    if(!evaluator.SetUp() || !evaluator.Evaluate())return 1;
    cv::FileStorage state{(output/"ablation_state.json").string(),cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON};
    state<<"variant"<<args.at("--variant")<<"flags"<<flags<<"prediction_smoothing_independent_of_prior"<<1
         <<"cad_matching_enabled"<<int(cad)<<"cad_renderer_created"<<int(cad)<<"video_orb"<<1
         <<"video_only_original_accumulation"<<int(!cad)
         <<"extra_recovery_enabled"<<0<<"input_file"<<args.at("--input");
    std::cout<<"Ablation complete "<<args.at("--variant")<<" D/F/P/T/N="<<flags<<std::endl;
  }catch(const std::exception &e){std::cerr<<"Ablation failed: "<<e.what()<<std::endl;return 1;}
}
