// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include "rbot_evaluator.h"

#include <algorithm>

int main(int argc, char *argv[]) {
  // Directories
  std::filesystem::path dataset_directory{R"(D:\datasets\RBOT_dataset)"};
  std::filesystem::path external_directory{
      L"D:\\workspace\\_cpp\\3DObjectTracking\\\u4e2d\u95f4\u7ed3\u679c\u6587\u4ef6\\RBOT"};
  std::filesystem::path result_directory{external_directory / "results"};
  std::filesystem::path optimization_directory{
      L"D:\\workspace\\_cpp\\3DObjectTracking\\\u4e2d\u95f4\u7ed3\u679c\u6587\u4ef6\\"
      L"RBOT_paper_optimization"};

  // Dataset configuration
  std::vector<std::string> body_names{
      "ape",  "bakingsoda", "benchviseblue", "broccolisoup", "cam",
      "can",  "cat",        "clown",         "cube",         "driller",
      "duck", "eggbox",     "glue",          "iron",         "koalacandy",
      "lamp", "phone",      "squirrel"};
  std::vector<std::string> sequence_names{
      "a_regular", "b_dynamiclight", "c_noisy", "d_occlusion", "d_occlusion"};
  std::vector<bool> sequence_occlusions{false, false, false, false, true};

  // Optional command line selection. Positional arguments remain compatible
  // with the original demo. Experiment names are deliberately limited to one
  // path component so all generated files stay below optimization_directory.
  std::vector<std::string> positional_arguments;
  std::string experiment_name;
  bool collect_diagnostics{false};
  bool run_sequentially{false};
  bool use_constant_velocity_prediction{false};
  bool use_noise_adaptive_filtering{false};
  bool use_gaussian_noise_filter{false};
  bool use_unfiltered_texture_observations{false};
  bool use_visibility_consensus{false};
  bool visibility_memory_only{false};
  bool visibility_local_memory{false};
  bool use_optical_flow_tracking{false};
  bool use_region_modality{true};
  bool use_texture_modality{true};
  bool use_sift_descriptor{false};
  bool use_reliability_weighted_fusion{false};
  bool use_keyframe_bank{false};
  bool use_directional_temporal_prior{false};
  bool use_windowed_temporal_prior{false};
  bool use_phase_consistent_temporal_prior{false};
  bool use_geometry_guided_matching{false};
  bool use_selective_recovery{false};
  bool use_validated_prediction_recovery{false};
  bool use_dual_model_texture_hypothesis{false};
  bool use_dual_hypothesis_independent_benchmark_reset{false};
  bool use_model_texture_matching{false};
  bool use_model_texture_sift{false};
  bool use_model_texture_akaze{false};
  bool use_model_texture_disk{false};
  bool use_adaptive_model_texture_disk{false};
  bool use_model_texture_disk_fusion{false};
  std::filesystem::path model_texture_disk_path{
      R"(D:\workspace\_env\m3t_lightglue\disk_512.onnx)"};
  float model_texture_disk_ransac_threshold{3.0f};
  int model_texture_disk_input_size{128};
  bool use_adaptive_model_texture_geometry{false};
  bool use_model_texture_pnp_geometry{false};
  bool use_model_photometric_tracking{false};
  bool use_model_contour_tracking{false};
  bool use_model_ecc_tracking{false};
  bool use_model_hypothesis_search{false};
  bool use_video_ecc_tracking{false};
  for (int i = 1; i < argc; ++i) {
    const std::string argument{argv[i]};
    if (argument == "--experiment") {
      if (++i >= argc) {
        std::cerr << "Missing experiment name after --experiment" << std::endl;
        return 1;
      }
      experiment_name = argv[i];
      if (experiment_name.empty() || experiment_name == "." ||
          experiment_name == ".." ||
          experiment_name.find_first_of("/\\") != std::string::npos) {
        std::cerr << "Experiment name must be one path component" << std::endl;
        return 1;
      }
    } else if (argument == "--diagnostics") {
      collect_diagnostics = true;
    } else if (argument == "--sequential") {
      run_sequentially = true;
    } else if (argument == "--constant-velocity") {
      use_constant_velocity_prediction = true;
    } else if (argument == "--noise-aware") {
      use_noise_adaptive_filtering = true;
    } else if (argument == "--visibility-aware") {
      use_visibility_consensus = true;
    } else if (argument == "--visibility-memory") {
      use_visibility_consensus = true;
      visibility_memory_only = true;
    } else if (argument == "--visibility-memory-local") {
      use_visibility_consensus = true;
      visibility_memory_only = true;
      visibility_local_memory = true;
    } else if (argument == "--noise-gaussian") {
      use_noise_adaptive_filtering = true;
      use_gaussian_noise_filter = true;
    } else if (argument == "--noise-region-only") {
      use_noise_adaptive_filtering = true;
      use_gaussian_noise_filter = true;
      use_unfiltered_texture_observations = true;
    } else if (argument == "--optical-flow") {
      use_optical_flow_tracking = true;
    } else if (argument == "--region-only") {
      use_texture_modality = false;
    } else if (argument == "--texture-only") {
      use_region_modality = false;
    } else if (argument == "--sift") {
      use_sift_descriptor = true;
    } else if (argument == "--reliability-fusion") {
      use_reliability_weighted_fusion = true;
    } else if (argument == "--keyframe-bank") {
      use_keyframe_bank = true;
    } else if (argument == "--directional-temporal-prior") {
      use_directional_temporal_prior = true;
    } else if (argument == "--windowed-temporal-prior") {
      use_windowed_temporal_prior = true;
    } else if (argument == "--phase-consistent-temporal-prior") {
      use_phase_consistent_temporal_prior = true;
    } else if (argument == "--geometry-guided-matching") {
      use_geometry_guided_matching = true;
    } else if (argument == "--selective-recovery") {
      use_selective_recovery = true;
    } else if (argument == "--validated-prediction-recovery") {
      use_validated_prediction_recovery = true;
    } else if (argument == "--dual-model-texture-hypothesis") {
      use_dual_model_texture_hypothesis = true;
    } else if (argument == "--dual-hypothesis-independent-benchmark-reset") {
      use_dual_hypothesis_independent_benchmark_reset = true;
    } else if (argument == "--model-texture") {
      use_model_texture_matching = true;
    } else if (argument == "--model-sift") {
      use_model_texture_matching = true;
      use_model_texture_sift = true;
    } else if (argument == "--model-akaze") {
      use_model_texture_matching = true;
      use_model_texture_akaze = true;
    } else if (argument == "--model-disk") {
      use_model_texture_matching = true;
      use_model_texture_disk = true;
    } else if (argument == "--adaptive-model-disk") {
      use_model_texture_matching = true;
      use_adaptive_model_texture_disk = true;
    } else if (argument == "--model-disk-fusion") {
      use_model_texture_matching = true;
      use_model_texture_disk_fusion = true;
    } else if (argument == "--disk-model") {
      if (++i >= argc) {
        std::cerr << "Missing model path after --disk-model" << std::endl;
        return 1;
      }
      model_texture_disk_path = argv[i];
    } else if (argument == "--disk-input-size") {
      if (++i >= argc) {
        std::cerr << "Missing value after --disk-input-size" << std::endl;
        return 1;
      }
      model_texture_disk_input_size = std::stoi(argv[i]);
    } else if (argument == "--disk-ransac-threshold") {
      if (++i >= argc) {
        std::cerr << "Missing value after --disk-ransac-threshold"
                  << std::endl;
        return 1;
      }
      model_texture_disk_ransac_threshold = std::stof(argv[i]);
    } else if (argument == "--adaptive-disk-geometry") {
      use_adaptive_model_texture_geometry = true;
    } else if (argument == "--disk-pnp-geometry") {
      use_model_texture_pnp_geometry = true;
    } else if (argument == "--model-photometric") {
      use_model_photometric_tracking = true;
    } else if (argument == "--model-contour") {
      use_model_contour_tracking = true;
    } else if (argument == "--model-ecc") {
      use_model_ecc_tracking = true;
    } else if (argument == "--model-hypothesis") {
      use_model_hypothesis_search = true;
    } else if (argument == "--video-ecc") {
      use_video_ecc_tracking = true;
    } else {
      positional_arguments.push_back(argument);
    }
  }

  if (positional_arguments.size() > 3) {
    std::cerr << "Usage: evaluate_rbot_dataset [body|all] "
                 "[a_regular|b_dynamiclight|c_noisy|d_occlusion|"
                 "d_occlusion_modeled|first4|all] [visualize] "
                 "[--experiment NAME] [--diagnostics] [--sequential] "
                 "[--constant-velocity] [--noise-aware] [--noise-gaussian]"
                 " [--noise-region-only]"
                 " [--visibility-aware] [--visibility-memory]"
                 " [--visibility-memory-local]"
                 " [--optical-flow]"
                 " [--region-only|--texture-only]"
                 " [--sift]"
                 " [--reliability-fusion]"
                 " [--keyframe-bank]"
                 " [--directional-temporal-prior]"
                 " [--windowed-temporal-prior]"
                 " [--phase-consistent-temporal-prior]"
                 " [--geometry-guided-matching]"
                 " [--selective-recovery]"
                 " [--validated-prediction-recovery]"
                 " [--dual-model-texture-hypothesis]"
                 " [--dual-hypothesis-independent-benchmark-reset]"
                 " [--model-texture]"
                 " [--model-sift]"
                 " [--model-akaze]"
                 " [--model-disk] [--disk-model PATH]"
                 " [--disk-input-size PX]"
                 " [--disk-ransac-threshold PX]"
                 " [--adaptive-disk-geometry]"
                 " [--disk-pnp-geometry]"
                 " [--adaptive-model-disk]"
                 " [--model-disk-fusion]"
                 " [--model-photometric]"
                 " [--model-contour]"
                 " [--model-ecc]"
                 " [--model-hypothesis]"
                 " [--video-ecc]"
              << std::endl;
    return 1;
  }
  if (use_dual_model_texture_hypothesis && !use_model_texture_disk) {
    std::cerr << "--dual-model-texture-hypothesis requires --model-disk"
              << std::endl;
    return 1;
  }
  if (use_dual_hypothesis_independent_benchmark_reset &&
      !use_dual_model_texture_hypothesis) {
    std::cerr << "--dual-hypothesis-independent-benchmark-reset requires "
                 "--dual-model-texture-hypothesis"
              << std::endl;
    return 1;
  }
  if (use_dual_model_texture_hypothesis &&
      (use_selective_recovery || use_validated_prediction_recovery)) {
    std::cerr << "Dual model-texture hypothesis cannot be combined with "
                 "prediction-recovery experiments"
              << std::endl;
    return 1;
  }
  if (!positional_arguments.empty() && positional_arguments[0] != "all") {
    const std::string selected_body{positional_arguments[0]};
    if (std::find(body_names.begin(), body_names.end(), selected_body) ==
        body_names.end()) {
      std::cerr << "Unknown RBOT body: " << selected_body << std::endl;
      return 1;
    }
    body_names = {selected_body};
  }
  if (positional_arguments.size() >= 2 && positional_arguments[1] == "first4") {
    sequence_names = {"a_regular", "b_dynamiclight", "c_noisy", "d_occlusion"};
    sequence_occlusions = {false, false, false, false};
  } else if (positional_arguments.size() >= 2 &&
             positional_arguments[1] != "all") {
    const std::string selected_sequence{positional_arguments[1]};
    const bool modeled_occlusion{selected_sequence == "d_occlusion_modeled"};
    const std::string sequence_name{
        modeled_occlusion ? "d_occlusion" : selected_sequence};
    const std::vector<std::string> valid_sequence_names{
        "a_regular", "b_dynamiclight", "c_noisy", "d_occlusion"};
    if (std::find(valid_sequence_names.begin(), valid_sequence_names.end(),
                  sequence_name) == valid_sequence_names.end()) {
      std::cerr << "Unknown RBOT sequence: " << selected_sequence << std::endl;
      return 1;
    }
    sequence_names = {sequence_name};
    sequence_occlusions = {modeled_occlusion};
  }
  bool visualize{false};
  if (positional_arguments.size() == 3) {
    if (positional_arguments[2] != "visualize") {
      std::cerr << "Unknown option: " << positional_arguments[2] << std::endl;
      return 1;
    }
    visualize = true;
  }

  std::filesystem::path diagnostics_directory;
  if (!experiment_name.empty()) {
    const auto experiment_directory{optimization_directory / experiment_name};
    result_directory = experiment_directory / "results";
    diagnostics_directory = experiment_directory / "diagnostics";
  } else if (collect_diagnostics) {
    std::cerr << "--diagnostics requires --experiment NAME" << std::endl;
    return 1;
  }

  std::filesystem::create_directories(result_directory);

  // Run experiments
  RBOTEvaluator evaluator{"evaluator", dataset_directory, external_directory,
                          body_names,  sequence_names,    sequence_occlusions};
  evaluator.set_region_modality_setter([&](auto r) {
    r->set_use_visibility_consensus(use_visibility_consensus);
    r->set_visibility_weight_contours(!visibility_memory_only);
    r->set_visibility_local_memory(visibility_local_memory);
    r->set_n_lines_max(200);
    r->set_use_adaptive_coverage(false);
    r->set_min_continuous_distance(3.0f);
    r->set_function_length(8);
    r->set_distribution_length(12);
    r->set_function_amplitude(0.36f);
    r->set_function_slope(0.0f);
    r->set_learning_rate(1.3f);
    r->set_scales({5, 2, 2, 1});
    r->set_standard_deviations({20.0f, 7.0f, 3.0f, 1.5f});
    r->set_n_histogram_bins(32);
    r->set_learning_rate_f(0.2f);
    r->set_learning_rate_b(0.2f);
    r->set_unconsidered_line_length(0.5f);
    r->set_max_considered_line_length(20.0f);
    r->set_modeled_depth_offset_radius(0.0f);
    r->set_modeled_occlusion_radius(0.0f);
    r->set_modeled_occlusion_threshold(0.03f);
  });
  evaluator.set_texture_modality_setter([&](auto t) {
    t->set_use_unfiltered_observations(use_unfiltered_texture_observations);
    t->set_descriptor_type(
        use_sift_descriptor ? m3t::TextureModality::DescriptorType::SIFT
                            : m3t::TextureModality::DescriptorType::ORB);
    t->set_focused_image_size(200);
    t->set_use_optical_flow_tracking(use_optical_flow_tracking);
    t->set_use_geometry_guided_matching(use_geometry_guided_matching);
    t->set_use_model_texture_sift(use_model_texture_sift);
    t->set_use_model_texture_akaze(use_model_texture_akaze);
    t->set_use_model_texture_disk(use_model_texture_disk,
                                  model_texture_disk_path);
    t->set_use_adaptive_model_texture_disk(use_adaptive_model_texture_disk,
                                           model_texture_disk_path);
    t->set_use_model_texture_disk_fusion(use_model_texture_disk_fusion,
                                         model_texture_disk_path);
    t->set_model_texture_disk_ransac_threshold(
        model_texture_disk_ransac_threshold);
    t->set_model_texture_disk_input_size(model_texture_disk_input_size);
    t->set_use_adaptive_model_texture_geometry(
        use_adaptive_model_texture_geometry);
    t->set_use_model_texture_pnp_geometry(use_model_texture_pnp_geometry);
    t->set_use_video_ecc_tracking(use_video_ecc_tracking);
    if (use_model_texture_matching || use_model_photometric_tracking ||
        use_model_contour_tracking || use_model_ecc_tracking ||
        use_model_hypothesis_search) {
      auto textured_renderer{
          std::make_shared<m3t::FocusedTexturedRenderer>(
              t->body_ptr()->name() + "_textured_renderer",
              t->silhouette_renderer_ptr()->renderer_geometry_ptr(),
              t->color_camera_ptr(), 200)};
      textured_renderer->AddReferencedBody(t->body_ptr());
      textured_renderer->SetUp();
      if (use_model_texture_matching)
        t->UseModelTextureMatching(textured_renderer);
      if (use_model_photometric_tracking)
        t->UseModelPhotometricTracking(textured_renderer);
      if (use_model_contour_tracking)
        t->UseModelContourTracking(textured_renderer);
      if (use_model_ecc_tracking)
        t->UseModelEccTracking(textured_renderer);
      if (use_model_hypothesis_search)
        t->UseModelHypothesisSearch(textured_renderer);
      if (!experiment_name.empty())
        t->set_model_texture_debug_directory(
            optimization_directory / experiment_name / "previews");
    }
    t->set_descriptor_distance_threshold(0.7f);
    t->set_tukey_norm_constant(20.0f);
    t->set_standard_deviations({5.0f, 1.0f});
    t->set_max_keyframe_rotation_difference(10.0f * m3t::kPi / 180.0f);
    t->set_max_keyframe_age(1000);
    t->set_n_keyframes(use_keyframe_bank ? 3 : 1);
    t->set_orb_n_features(300);
    t->set_orb_scale_factor(1.2f);
    t->set_orb_n_levels(3);
    t->set_brisk_threshold(35);
    t->set_brisk_octave(3);
    t->set_brisk_pattern_scale(0.6f);
    t->set_daisy_radius(8.0f);
    t->set_daisy_q_radius(3);
    t->set_daisy_q_theta(4);
    t->set_daisy_q_hist(8);
    t->set_freak_orientation_normalized(true);
    t->set_freak_scale_normalized(true);
    t->set_freak_pattern_scale(16.0f);
    t->set_freak_n_octaves(4);
    t->set_sift_n_features(0);
    t->set_sift_n_octave_layers(3);
    t->set_sift_contrast_threshold(0.04);
    t->set_sift_edge_threshold(10.0);
    t->set_sift_sigma(0.7f);
    t->set_modeled_occlusion_radius(0.0f);
    t->set_modeled_occlusion_threshold(0.03f);
  });
  evaluator.set_optimizer_setter([&](auto o) {
    o->set_tikhonov_parameter_rotation(1000.0f);
    o->set_tikhonov_parameter_translation(30000.0f);
  });
  evaluator.set_tracker_setter([&](auto t) {
    t->set_n_update_iterations(2);
    t->set_n_corr_iterations(7);
  });
  evaluator.set_run_sequentially(run_sequentially);
  evaluator.set_use_noise_adaptive_filtering(use_noise_adaptive_filtering);
  evaluator.set_use_gaussian_noise_filter(use_gaussian_noise_filter);
  evaluator.set_visualize_all_results(visualize);
  evaluator.set_use_constant_velocity_prediction(
      use_constant_velocity_prediction);
  evaluator.set_use_reliability_weighted_fusion(
      use_reliability_weighted_fusion);
  evaluator.set_use_directional_temporal_prior(
      use_directional_temporal_prior);
  evaluator.set_use_windowed_temporal_prior(use_windowed_temporal_prior);
  evaluator.set_use_phase_consistent_temporal_prior(
      use_phase_consistent_temporal_prior);
  evaluator.set_use_selective_recovery(use_selective_recovery);
  evaluator.set_use_validated_prediction_recovery(
      use_validated_prediction_recovery);
  evaluator.set_use_dual_model_texture_hypothesis(
      use_dual_model_texture_hypothesis);
  evaluator.set_use_dual_hypothesis_independent_benchmark_reset(
      use_dual_hypothesis_independent_benchmark_reset);
  if (!use_region_modality && !use_texture_modality) {
    std::cerr << "Cannot combine --region-only and --texture-only" << std::endl;
    return 1;
  }
  evaluator.set_use_region_modality(use_region_modality);
  evaluator.set_use_texture_modality(use_texture_modality);
  evaluator.SaveResults(result_directory);
  if (collect_diagnostics)
    evaluator.EnableDiagnostics(diagnostics_directory);
  if (!evaluator.SetUp()) return 1;
  if (!evaluator.Evaluate()) return 1;
  return 0;
}
