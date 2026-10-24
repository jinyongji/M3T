// SPDX-License-Identifier: MIT
// Industrial RGB adapter for the current RBOT final configuration.
#include "rbot_evaluator.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

int main(int argc, char **argv) {
#ifndef USE_ONNXRUNTIME
  std::cerr << "Industrial final requires -DUSE_ONNXRUNTIME=ON" << std::endl;
  return 1;
#endif
  try {
    bool benchmark_mode = false;
    std::map<std::string, std::string> args;
    const std::set<std::string> keys{"--dataset", "--cache", "--output", "--scene",
        "--motion", "--condition", "--object", "--frames", "--disk-model"};
    for (int i = 1; i < argc; ++i) {
      const std::string key{argv[i]};
      if (key == "--benchmark") { benchmark_mode = true; continue; }
      if (!keys.count(key) || ++i >= argc || args.count(key))
        throw std::runtime_error("Unknown, duplicate or missing argument: " + key);
      args[key] = argv[i];
    }
    for (const auto &key : keys)
      if (!args.count(key)) throw std::runtime_error("Required argument: " + key);
    const std::filesystem::path root{args.at("--dataset")};
    const std::filesystem::path output{args.at("--output")};
    const std::string scene{args.at("--scene")}, motion{args.at("--motion")},
        condition{args.at("--condition")}, object{args.at("--object")};
    const std::vector<std::string> scenes{"scene_01", "scene_02", "scene_03", "scene_04", "scene_05"};
    const std::vector<std::string> motions{"camera_motion", "camera_and_objects_motion"};
    const std::vector<std::string> conditions{"01_normal", "02_low_light", "03_hard_glare", "04_fast_motion", "05_sensor_noise"};
    const auto contains = [](const auto &v, const auto &s) {
      return std::find(v.begin(), v.end(), s) != v.end();
    };
    if (!contains(scenes, scene) || !contains(motions, motion) || !contains(conditions, condition))
      throw std::runtime_error("Invalid scene, motion or condition");
    const int scene_id = int(std::find(scenes.begin(), scenes.end(), scene) - scenes.begin()) + 1;
    bool valid_object = false;
    for (int id = scene_id * 3 - 2; id <= scene_id * 3; ++id) {
      std::ostringstream expected;
      expected << "obj_" << std::setw(6) << std::setfill('0') << id;
      valid_object = valid_object || expected.str() == object;
    }
    if (!valid_object) throw std::runtime_error("Object is not assigned to the selected scene");
    size_t consumed = 0;
    const int frames = std::stoi(args.at("--frames"), &consumed);
    if (consumed != args.at("--frames").size() || frames < 2 || frames > 500)
      throw std::runtime_error("--frames must be an integer from 2 to 500, including initialization");

    cv::FileStorage camera{(root / "cameras" / "camera.json").string(), cv::FileStorage::READ};
    if (!camera.isOpened()) throw std::runtime_error("Cannot open cameras/camera.json");
    const auto k = camera["K"];
    if (!k.isSeq() || k.size() != 3) throw std::runtime_error("Expected nested 3x3 K");
    for (int r = 0; r < 3; ++r)
      if (!k[r].isSeq() || k[r].size() != 3) throw std::runtime_error("Expected nested 3x3 K");
    if (int(camera["width"]) != 900 || int(camera["height"]) != 600 ||
        int(camera["frames_per_sequence"]) != 500 ||
        std::string(camera["translation_unit"]) != "meter" ||
        std::string(camera["pose_direction"]) != "p_camera = R @ p_object + t")
      throw std::runtime_error("This adapter requires the 900x600 v0.8 metre-valued dataset");
    const double expected_k[3][3]{{1125, 0, 449.5}, {0, 1125, 299.5}, {0, 0, 1}};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        if (!std::isfinite(double(k[r][c])) || std::abs(double(k[r][c]) - expected_k[r][c]) > 1e-6)
          throw std::runtime_error("Unexpected intrinsics; do not silently rescale the input");
    for (const auto &v : camera["distortion_coefficients"])
      if (double(v) != 0) throw std::runtime_error("Distorted images require a separate adapter");

    RBOTEvaluator::ExternalSequence input;
    input.image_directory = root / "images" / scene / motion / condition;
    input.model_file = root / "objects" / object / (object + ".obj");
    input.pose_file = root / "poses" / scene / motion / condition / ("poses_" + object.substr(4) + ".txt");
    input.intrinsics = {float(double(k[0][0])), float(double(k[1][1])),
        float(double(k[0][2])), float(double(k[1][2])), 900, 600};
    input.image_count = frames;
    if (!std::filesystem::is_regular_file(input.model_file) ||
        !std::filesystem::is_regular_file(args.at("--disk-model")))
      throw std::runtime_error("CAD model or DISK ONNX file is missing");
    const auto sequence = scene + "__" + motion + "__" + condition;
    RBOTEvaluator evaluator{"industrial", root, args.at("--cache"), {object}, {sequence}, {false}};
    evaluator.set_external_sequence(input);
    evaluator.set_region_modality_setter([](auto r) {
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
      t->set_use_unfiltered_observations(true);
      t->set_descriptor_type(m3t::TextureModality::DescriptorType::ORB);
      t->set_focused_image_size(200);
      t->set_use_model_texture_disk(true, args.at("--disk-model"));
      t->set_model_texture_disk_input_size(128);
      t->set_model_texture_disk_ransac_threshold(3.0f);
      auto renderer = std::make_shared<m3t::FocusedTexturedRenderer>(
          object + "_textured_renderer", t->silhouette_renderer_ptr()->renderer_geometry_ptr(),
          t->color_camera_ptr(), 200);
      renderer->AddReferencedBody(t->body_ptr());
      if (!renderer->SetUp()) throw std::runtime_error("Textured renderer setup failed");
      t->UseModelTextureMatching(renderer);
      t->set_descriptor_distance_threshold(0.7f);
      t->set_tukey_norm_constant(20.0f);
      t->set_standard_deviations({5.0f, 1.0f});
      t->set_max_keyframe_rotation_difference(10.0f * m3t::kPi / 180.0f);
      t->set_max_keyframe_age(1000);
      t->set_n_keyframes(1);
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
    evaluator.set_optimizer_setter([](auto o) {
      o->set_tikhonov_parameter_rotation(1000.0f);
      o->set_tikhonov_parameter_translation(30000.0f);
    });
    evaluator.set_tracker_setter([](auto t) {
      t->set_n_update_iterations(2);
      t->set_n_corr_iterations(7);
    });
    evaluator.set_run_sequentially(true);
    evaluator.set_visualize_all_results(false);
    evaluator.set_use_region_modality(true);
    evaluator.set_use_texture_modality(true);
    evaluator.set_use_noise_adaptive_filtering(true);
    evaluator.set_use_gaussian_noise_filter(true);
    evaluator.set_use_constant_velocity_prediction(true);
    evaluator.set_use_reliability_weighted_fusion(true);
    evaluator.set_use_directional_temporal_prior(true);
    evaluator.SaveResults(output / "results");
    if (!benchmark_mode) evaluator.EnableDiagnostics(output / "diagnostics");
    if (benchmark_mode) cv::setNumThreads(1);
    std::cout << "Industrial final: " << scene << '/' << motion << '/' << condition
              << '/' << object << "; initialized image 1; evaluating images 2.." << frames
              << "; metre scale 1; 16-bit RGB / 257; GT reset at 5cm/5deg" << std::endl;
    if (!evaluator.SetUp() || !evaluator.Evaluate()) return 1;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Industrial evaluation failed: " << e.what() << std::endl;
    return 1;
  }
}
