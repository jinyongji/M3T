// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef M3T_EXAMPLES_RBOT_EVALUATOR_H_
#define M3T_EXAMPLES_RBOT_EVALUATOR_H_

#include <m3t/basic_depth_renderer.h>
#include <m3t/body.h>
#include <m3t/common.h>
#include <m3t/link.h>
#include <m3t/loader_camera.h>
#include <m3t/modality.h>
#include <m3t/normal_viewer.h>
#include <m3t/optimizer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/renderer_geometry.h>
#include <m3t/texture_modality.h>
#include <m3t/tracker.h>

#include <filesystem/filesystem.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * \brief Class that holds a tracker to evaluate it on the RBOT dataset.
 * \details It includes functionality to reset bodies, calculate rotational and
 * translational errors, compute tracking success, measure execution times, and
 * save results to a file
 */
class RBOTEvaluator {
 private:
  static constexpr int kNFrames_ = 1000;
  static constexpr m3t::Intrinsics kRBOTIntrinsics{
      650.048f, 647.183f, 324.328f - 0.5f, 257.323f - 0.5f, 640, 512};
  const std::string kOcclusionBodyName{"squirrel_small"};

  struct RunConfiguration {
    std::string sequence_name{};
    bool occlusions;
    std::string body_name{};
  };

  struct ExecutionTimes {
    float complete_cycle = 0.0f;
    float calculate_correspondences = 0.0f;
    float calculate_gradient_and_hessian = 0.0f;
    float calculate_optimization = 0.0f;
    float calculate_results = 0.0f;
  };

  struct ModalityDiagnostics {
    int n_correspondences = 0;
    int n_gradient_terms = 0;
    float residual_mean = 0.0f;
    float inlier_fraction = 0.0f;
    float auxiliary_mean = 0.0f;
    float normalized_gradient_norm = 0.0f;
    float normalized_information_trace = 0.0f;
    float normalized_information_min_eigenvalue = 0.0f;
    float normalized_information_max_eigenvalue = 0.0f;
    float normalized_information_condition = 0.0f;
    float step_rotation_norm = 0.0f;
    float step_translation_norm = 0.0f;
    Eigen::Matrix<float, 6, 1> regularized_step =
        Eigen::Matrix<float, 6, 1>::Zero();
  };

  struct FusionDiagnostics {
    float image_noise_sigma = 0.0f;
    float image_preprocessing_us = 0.0f;
    float region_visibility_evidence = 1.0f;
    float region_visibility_activation = 0.0f;
    ModalityDiagnostics region;
    ModalityDiagnostics texture;
    float step_cosine = 0.0f;
    float regularized_step_disagreement = 0.0f;
    int model_texture_correspondences = 0;
    float model_texture_residual_mean = 0.0f;
    float model_texture_reliability_weight = 0.0f;
    int model_photometric_points = 0;
    float model_photometric_residual_mean = 0.0f;
    float model_photometric_reliability_weight = 0.0f;
    int model_contour_points = 0;
    float model_contour_residual_mean = 0.0f;
    float model_contour_reliability_weight = 0.0f;
    int model_ecc_points = 0;
    float model_ecc_residual_mean = 0.0f;
    float model_ecc_correlation = 0.0f;
    float model_ecc_reliability_weight = 0.0f;
    float model_hypothesis_confidence = 0.0f;
    float model_hypothesis_reliability_weight = 0.0f;
    int video_ecc_points = 0;
    float video_ecc_residual_mean = 0.0f;
    float video_ecc_correlation = 0.0f;
    float video_ecc_reliability_weight = 0.0f;
    bool selective_recovery = false;
    float dual_hypothesis_log_odds = 0.0f;
    bool dual_hypothesis_selected_disk = false;
    float dual_hypothesis_disk_success = 0.0f;
    float dual_hypothesis_orb_success = 0.0f;
  };

  struct Result {
    int frame_index = 0;
    // Final output before any benchmark ground-truth reset, not the next
    // frame's initial state. Keep native world poses until serialization.
    m3t::Transform3fA estimated_pose{m3t::Transform3fA::Identity()};
    m3t::Transform3fA ground_truth_pose{m3t::Transform3fA::Identity()};
    bool reset_primary_after_frame = false;
    bool reset_secondary_after_frame = false;
    float translation_error = 0.0f;
    float rotation_error = 0.0f;
    float tracking_success = 0.0f;
    bool prediction_valid = false;
    float prediction_translation_error = 0.0f;
    float prediction_rotation_error = 0.0f;
    float prediction_success = 0.0f;
    ExecutionTimes execution_times;
    FusionDiagnostics fusion_diagnostics;
  };

 public:
  // Constructors and setup method
  RBOTEvaluator(const std::string &name,
                const std::filesystem::path &dataset_directory,
                const std::filesystem::path &external_directory,
                const std::vector<std::string> &body_names,
                const std::vector<std::string> &sequence_names,
                const std::vector<bool> &sequence_occlusions);
  bool SetUp();

  // Setters for evaluation
  void set_translation_error_threshold(float translation_error_threshold);
  void set_rotation_error_threshold(float rotation_error_threshold);
  void set_run_sequentially(bool run_sequentially);
  void set_visualize_all_results(bool visualize_all_results);
  void set_use_noise_adaptive_filtering(bool enabled) {
    use_noise_adaptive_filtering_ = enabled;
  }
  void set_use_gaussian_noise_filter(bool enabled) { use_gaussian_noise_filter_ = enabled; }
  void set_use_constant_velocity_prediction(
      bool use_constant_velocity_prediction);
  void set_use_reliability_weighted_fusion(
      bool use_reliability_weighted_fusion);
  void set_use_directional_temporal_prior(bool use_directional_temporal_prior);
  void set_use_windowed_temporal_prior(bool use_windowed_temporal_prior);
  void set_use_phase_consistent_temporal_prior(
      bool use_phase_consistent_temporal_prior);
  void set_use_selective_recovery(bool use_selective_recovery);
  void set_use_validated_prediction_recovery(
      bool use_validated_prediction_recovery);
  void set_use_dual_model_texture_hypothesis(
      bool use_dual_model_texture_hypothesis);
  void set_use_dual_hypothesis_independent_benchmark_reset(
      bool use_dual_hypothesis_independent_benchmark_reset);
  void SaveResults(std::filesystem::path save_directory);
  void DoNotSaveResults();
  void EnableDiagnostics(std::filesystem::path diagnostics_directory);
  void DisableDiagnostics();

  // Setters for tracker configuration
  void set_use_region_modality(bool use_region_modality);
  void set_use_texture_modality(bool use_texture_modality);

  // Setters for object setters
  void set_tracker_setter(
      const std::function<void(std::shared_ptr<m3t::Tracker>)> &tracker_setter);
  void set_optimizer_setter(
      const std::function<void(std::shared_ptr<m3t::Optimizer>)>
          &optimizer_setter);
  void set_region_modality_setter(
      const std::function<void(std::shared_ptr<m3t::RegionModality>)>
          &region_modality_setter);
  void set_region_model_setter(
      const std::function<void(std::shared_ptr<m3t::RegionModel>)>
          &region_model_setter);
  void set_texture_modality_setter(
      const std::function<void(std::shared_ptr<m3t::TextureModality>)>
          &texture_modality_setter);
  void set_depth_renderer_setter(
      const std::function<void(std::shared_ptr<m3t::FocusedBasicDepthRenderer>)>
          &depth_renderer_setter);

  // Main methods
  bool Evaluate();

  // Getters
  float tracking_success() const;
  float execution_time() const;

 private:
  // Helper methods to run evaluation
  bool EvaluateRunConfiguration(
      const RunConfiguration &run_configuration,
      const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
      Result *average_result) const;
  void SetUpTracker(
      const RunConfiguration &run_configuration,
      const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
      std::shared_ptr<m3t::Tracker> *tracker_ptr,
      bool force_orb_model_texture = false) const;
  void ResetBody(const std::shared_ptr<m3t::Tracker> &tracker_ptr,
                 int i_frame) const;
  void ResetOcclusionBody(const std::shared_ptr<m3t::Tracker> &tracker_ptr,
                          int i_frame) const;
  void ExecuteMeasuredTrackingCycle(
      const std::shared_ptr<m3t::Tracker> &tracker_ptr, int iteration,
      ExecutionTimes *execution_times, bool calculate_results = true) const;
  void CollectDiagnostics(const std::shared_ptr<m3t::Tracker> &tracker_ptr,
                          FusionDiagnostics *diagnostics) const;
  static ModalityDiagnostics CalculateModalityDiagnostics(
      const std::shared_ptr<m3t::Modality> &modality_ptr,
      float tikhonov_parameter_rotation,
      float tikhonov_parameter_translation);

  // Helper methods for the calculation of results
  void CalculatePoseResults(const m3t::Transform3fA &body2world_pose,
                            const m3t::Transform3fA &body2world_pose_gt,
                            Result *result) const;
  static void CalculateAverageResult(const std::vector<Result> &results,
                                     Result *average_result);
  static void VisualizeFrameResult(const Result &result,
                                   const std::string &title);
  static void VisualizeFinalResult(const Result &result,
                                   const std::string &title);
  void SaveFinalResult(const Result &result, const std::string &title) const;
  void SaveDiagnostics(const std::vector<Result> &results,
                       const std::string &title) const;
  bool SaveFramePoses(const std::vector<Result> &results,
                      const RunConfiguration &configuration,
                      const std::shared_ptr<m3t::Tracker> &tracker) const;

  // General helper methods
  bool LoadBodies();
  bool LoadSingleBody(const std::string &body_name,
                      const std::filesystem::path &directory);
  bool GenerateModels();
  bool GenerateSingleModel(const std::string &body_name,
                           const std::filesystem::path &directory);
  static bool ReadPosesRBOTDataset(const std::filesystem::path &path,
                                   std::vector<m3t::Transform3fA> *poses);
  static float ElapsedTime(
      const std::chrono::high_resolution_clock::time_point &begin_time);

  // Internal data objects
  std::map<std::string, std::shared_ptr<m3t::Body>> body2body_ptr_map_;
  std::map<std::string, std::shared_ptr<m3t::RegionModel>> body2model_ptr_map_;
  std::vector<m3t::Transform3fA> poses_gt_first_;
  std::vector<m3t::Transform3fA> poses_gt_second_;
  std::vector<RunConfiguration> run_configurations_;
  std::vector<Result> results_;
  Result final_result_;

  // Parameters for RBOT dataset
  std::filesystem::path dataset_directory_;
  std::filesystem::path external_directory_;
  std::vector<std::string> body_names_;
  std::vector<std::string> sequence_names_;
  std::vector<bool> sequence_occlusions_;

  // Parameters for tracker configuration
  bool use_region_modality_;
  bool use_texture_modality_;

  // Setters for object setters
  std::function<void(std::shared_ptr<m3t::Tracker>)> tracker_setter_{
      [](auto) {}};
  std::function<void(std::shared_ptr<m3t::Optimizer>)> optimizer_setter_{
      [](auto) {}};
  std::function<void(std::shared_ptr<m3t::RegionModality>)>
      region_modality_setter_{[](auto) {}};
  std::function<void(std::shared_ptr<m3t::RegionModel>)> region_model_setter_{
      [](auto) {}};
  std::function<void(std::shared_ptr<m3t::TextureModality>)>
      texture_modality_setter_{[](auto) {}};
  std::function<void(std::shared_ptr<m3t::FocusedBasicDepthRenderer>)>
      depth_renderer_setter_{[](auto) {}};

  // Parameters for evaluator
  std::string name_{};
  float translation_error_threshold_ = 0.05f;
  float rotation_error_threshold_ = 5.0f * m3t::kPi / 180.0f;
  bool run_sequentially_ = true;
  bool visualize_all_results_ = true;
  bool use_constant_velocity_prediction_ = false;
  bool use_noise_adaptive_filtering_ = false;
  bool use_gaussian_noise_filter_ = false;
  bool use_reliability_weighted_fusion_ = false;
  bool use_directional_temporal_prior_ = false;
  bool use_windowed_temporal_prior_ = false;
  bool use_phase_consistent_temporal_prior_ = false;
  bool use_selective_recovery_ = false;
  bool use_validated_prediction_recovery_ = false;
  bool use_dual_model_texture_hypothesis_ = false;
  bool use_dual_hypothesis_independent_benchmark_reset_ = false;
  bool save_results_ = false;
  std::filesystem::path save_directory_;
  bool collect_diagnostics_ = false;
  std::filesystem::path diagnostics_directory_;
  bool set_up_ = false;
};

#endif  // M3T_EXAMPLES_RBOT_EVALUATOR_H_
