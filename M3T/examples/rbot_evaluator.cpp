// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include "rbot_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <limits>

RBOTEvaluator::RBOTEvaluator(const std::string &name,
                             const std::filesystem::path &dataset_directory,
                             const std::filesystem::path &external_directory,
                             const std::vector<std::string> &body_names,
                             const std::vector<std::string> &sequence_names,
                             const std::vector<bool> &sequence_occlusions)
    : name_{name},
      dataset_directory_{dataset_directory},
      external_directory_{external_directory},
      body_names_{body_names},
      sequence_names_{sequence_names},
      sequence_occlusions_{sequence_occlusions} {}

bool RBOTEvaluator::SetUp() {
  set_up_ = false;

  // Create run configurations
  run_configurations_.clear();
  for (size_t i = 0; i < sequence_names_.size(); ++i) {
    for (size_t j = 0; j < body_names_.size(); ++j) {
      run_configurations_.push_back(RunConfiguration{
          sequence_names_[i], sequence_occlusions_[i], body_names_[j]});
    }
  }

  // Read poses
  if (!ReadPosesRBOTDataset(dataset_directory_ / "poses_first.txt",
                            &poses_gt_first_))
    return false;
  if (!ReadPosesRBOTDataset(dataset_directory_ / "poses_second.txt",
                            &poses_gt_second_))
    return false;

  // Load bodies and generate models
  LoadBodies();
  GenerateModels();

  set_up_ = true;
  return true;
}

void RBOTEvaluator::set_translation_error_threshold(
    float translation_error_threshold) {
  translation_error_threshold_ = translation_error_threshold;
}

void RBOTEvaluator::set_rotation_error_threshold(
    float rotation_error_threshold) {
  rotation_error_threshold_ = rotation_error_threshold;
}

void RBOTEvaluator::set_run_sequentially(bool run_sequentially) {
  run_sequentially_ = run_sequentially;
}

void RBOTEvaluator::set_visualize_all_results(bool visualize_all_results) {
  visualize_all_results_ = visualize_all_results;
}

void RBOTEvaluator::set_use_constant_velocity_prediction(
    bool use_constant_velocity_prediction) {
  use_constant_velocity_prediction_ = use_constant_velocity_prediction;
}

void RBOTEvaluator::set_use_reliability_weighted_fusion(
    bool use_reliability_weighted_fusion) {
  use_reliability_weighted_fusion_ = use_reliability_weighted_fusion;
}

void RBOTEvaluator::set_use_directional_temporal_prior(
    bool use_directional_temporal_prior) {
  use_directional_temporal_prior_ = use_directional_temporal_prior;
}

void RBOTEvaluator::set_use_windowed_temporal_prior(
    bool use_windowed_temporal_prior) {
  use_windowed_temporal_prior_ = use_windowed_temporal_prior;
}

void RBOTEvaluator::set_use_phase_consistent_temporal_prior(
    bool use_phase_consistent_temporal_prior) {
  use_phase_consistent_temporal_prior_ =
      use_phase_consistent_temporal_prior;
}

void RBOTEvaluator::set_use_selective_recovery(bool use_selective_recovery) {
  use_selective_recovery_ = use_selective_recovery;
}

void RBOTEvaluator::set_use_validated_prediction_recovery(
    bool use_validated_prediction_recovery) {
  use_validated_prediction_recovery_ = use_validated_prediction_recovery;
}

void RBOTEvaluator::set_use_dual_model_texture_hypothesis(
    bool use_dual_model_texture_hypothesis) {
  use_dual_model_texture_hypothesis_ = use_dual_model_texture_hypothesis;
}

void RBOTEvaluator::set_use_dual_hypothesis_independent_benchmark_reset(
    bool use_dual_hypothesis_independent_benchmark_reset) {
  use_dual_hypothesis_independent_benchmark_reset_ =
      use_dual_hypothesis_independent_benchmark_reset;
}

void RBOTEvaluator::SaveResults(std::filesystem::path save_directory) {
  save_directory_ = save_directory;
  save_results_ = true;
}

void RBOTEvaluator::DoNotSaveResults() { save_results_ = false; }

void RBOTEvaluator::EnableDiagnostics(
    std::filesystem::path diagnostics_directory) {
  diagnostics_directory_ = std::move(diagnostics_directory);
  std::filesystem::create_directories(diagnostics_directory_);
  collect_diagnostics_ = true;
}

void RBOTEvaluator::DisableDiagnostics() { collect_diagnostics_ = false; }

void RBOTEvaluator::set_use_region_modality(bool use_region_modality) {
  use_region_modality_ = use_region_modality;
}

void RBOTEvaluator::set_use_texture_modality(bool use_texture_modality) {
  use_texture_modality_ = use_texture_modality;
}

void RBOTEvaluator::set_tracker_setter(
    const std::function<void(std::shared_ptr<m3t::Tracker>)> &tracker_setter) {
  tracker_setter_ = tracker_setter;
}

void RBOTEvaluator::set_optimizer_setter(
    const std::function<void(std::shared_ptr<m3t::Optimizer>)>
        &optimizer_setter) {
  optimizer_setter_ = optimizer_setter;
}

void RBOTEvaluator::set_region_modality_setter(
    const std::function<void(std::shared_ptr<m3t::RegionModality>)>
        &region_modality_setter) {
  region_modality_setter_ = region_modality_setter;
}

void RBOTEvaluator::set_region_model_setter(
    const std::function<void(std::shared_ptr<m3t::RegionModel>)>
        &region_model_setter) {
  region_model_setter_ = region_model_setter;
  set_up_ = false;
}

void RBOTEvaluator::set_texture_modality_setter(
    const std::function<void(std::shared_ptr<m3t::TextureModality>)>
        &texture_modality_setter) {
  texture_modality_setter_ = texture_modality_setter;
}

void RBOTEvaluator::set_depth_renderer_setter(
    const std::function<void(std::shared_ptr<m3t::FocusedBasicDepthRenderer>)>
        &depth_renderer_setter) {
  depth_renderer_setter_ = depth_renderer_setter;
}

bool RBOTEvaluator::Evaluate() {
  if (!set_up_) {
    std::cerr << "Set up evaluator " << name_ << " first" << std::endl;
    return false;
  }

  // Evaluate all run configuration
  results_.resize(run_configurations_.size());
  if (run_sequentially_ || visualize_all_results_) {
    auto renderer_geometry_ptr{std::make_shared<m3t::RendererGeometry>("rg")};
    renderer_geometry_ptr->SetUp();
    for (size_t i = 0; i < int(run_configurations_.size()); ++i) {
      EvaluateRunConfiguration(run_configurations_[i], renderer_geometry_ptr,
                               &results_[i]);

      std::string title{run_configurations_[i].sequence_name + "_" +
                        (run_configurations_[i].occlusions ? "modeled_" : "") +
                        run_configurations_[i].body_name};
      if (save_results_) SaveFinalResult(results_[i], title);
      VisualizeFinalResult(results_[i], title);
    }
  } else {
    std::vector<std::shared_ptr<m3t::RendererGeometry>> renderer_geometry_ptrs(
        omp_get_max_threads());
    for (auto &renderer_geometry_ptr : renderer_geometry_ptrs) {
      renderer_geometry_ptr = std::make_shared<m3t::RendererGeometry>("rg");
      renderer_geometry_ptr->SetUp();
    }
#pragma omp parallel for
    for (int i = 0; i < int(run_configurations_.size()); ++i) {
      EvaluateRunConfiguration(run_configurations_[i],
                               renderer_geometry_ptrs[omp_get_thread_num()],
                               &results_[i]);

      std::string title{run_configurations_[i].sequence_name + "_" +
                        (run_configurations_[i].occlusions ? "modeled_" : "") +
                        run_configurations_[i].body_name};
      if (save_results_) SaveFinalResult(results_[i], title);
#pragma omp critical
      VisualizeFinalResult(results_[i], title);
    }
  }

  // Calculate final results
  CalculateAverageResult(results_, &final_result_);
  if (save_results_) SaveFinalResult(final_result_, "all_sequences_all_bodies");
  VisualizeFinalResult(final_result_, "all_sequences_all_bodies");
  return true;
}

float RBOTEvaluator::tracking_success() const {
  return final_result_.tracking_success;
}

float RBOTEvaluator::execution_time() const {
  return final_result_.execution_times.complete_cycle;
}

void RBOTEvaluator::EvaluateRunConfiguration(
    const RunConfiguration &run_configuration,
    const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
    Result *average_result) const {
  std::shared_ptr<m3t::Tracker> tracker_ptr;
  SetUpTracker(run_configuration, renderer_geometry_ptr, &tracker_ptr);
  const auto &body_ptr{tracker_ptr->optimizer_ptrs()[0]
                           ->root_link_ptr()
                           ->modality_ptrs()[0]
                           ->body_ptr()};
  std::shared_ptr<m3t::RendererGeometry> orb_renderer_geometry_ptr;
  std::shared_ptr<m3t::Tracker> orb_tracker_ptr;
  std::shared_ptr<m3t::Body> orb_body_ptr;
  if (use_dual_model_texture_hypothesis_) {
    orb_renderer_geometry_ptr =
        std::make_shared<m3t::RendererGeometry>("rg_orb_hypothesis");
    orb_renderer_geometry_ptr->SetUp();
    SetUpTracker(run_configuration, orb_renderer_geometry_ptr,
                 &orb_tracker_ptr, true);
    orb_body_ptr = orb_tracker_ptr->optimizer_ptrs()[0]
                       ->root_link_ptr()
                       ->body_ptr();
  }
  ResetBody(tracker_ptr, 0);
  if (orb_tracker_ptr) ResetBody(orb_tracker_ptr, 0);
  if (run_configuration.occlusions) ResetOcclusionBody(tracker_ptr, 0);

  m3t::Transform3fA previous_previous_pose{body_ptr->body2world_pose()};
  m3t::Transform3fA previous_pose{body_ptr->body2world_pose()};
  m3t::Transform3fA last_pose{body_ptr->body2world_pose()};
  std::deque<m3t::Transform3fA> successful_pose_history{
      body_ptr->body2world_pose()};
  bool prediction_ready{false};
  bool smoothed_prediction_ready{false};
  m3t::Transform3fA orb_previous_previous_pose{
      orb_body_ptr ? orb_body_ptr->body2world_pose()
                   : m3t::Transform3fA::Identity()};
  m3t::Transform3fA orb_previous_pose{orb_previous_previous_pose};
  m3t::Transform3fA orb_last_pose{orb_previous_previous_pose};
  bool orb_prediction_ready{false};
  bool orb_smoothed_prediction_ready{false};

  // Iterate over all frames
  std::vector<Result> results(kNFrames_);
  for (int i = 0; i < kNFrames_; ++i) {
    m3t::Transform3fA selected_pose_for_evaluation{
        body_ptr->body2world_pose()};
    m3t::Transform3fA predicted_pose_for_frame{m3t::Transform3fA::Identity()};
    bool predicted_pose_valid{false};
    if ((use_constant_velocity_prediction_ ||
         use_directional_temporal_prior_ || use_windowed_temporal_prior_ ||
         use_phase_consistent_temporal_prior_ ||
         use_selective_recovery_ || use_validated_prediction_recovery_) &&
        prediction_ready) {
      m3t::Transform3fA predicted_pose{
          last_pose * previous_pose.inverse() * last_pose};
      m3t::Transform3fA phase_prior_pose{m3t::Transform3fA::Identity()};
      bool phase_prior_pose_valid{false};
      if (use_phase_consistent_temporal_prior_ &&
          successful_pose_history.size() >= 6) {
        // Use phase-consistent velocities at lags 0, 2, and 4. Averaging
        // adjacent estimates can retain alternating pose jitter; sampling one
        // phase suppresses it while preserving a five-frame temporal span.
        const size_t history_size{successful_pose_history.size()};
        std::array<Eigen::Vector3f, 5> rotation_velocities;
        std::array<Eigen::Vector3f, 5> translation_velocities;
        for (size_t velocity_index = 0; velocity_index < 5;
             ++velocity_index) {
          const size_t current_index{history_size - 5 + velocity_index};
          const m3t::Transform3fA variation{
              successful_pose_history[current_index - 1].inverse() *
              successful_pose_history[current_index]};
          const Eigen::AngleAxisf rotation{variation.rotation().matrix()};
          if (std::isfinite(rotation.angle())) {
            rotation_velocities[velocity_index] =
                rotation.angle() * rotation.axis();
          } else {
            rotation_velocities[velocity_index].setZero();
          }
          translation_velocities[velocity_index] = variation.translation();
        }
        const Eigen::Vector3f phase_rotation{
            (rotation_velocities[0] + rotation_velocities[2] +
             rotation_velocities[4]) /
            3.0f};
        const Eigen::Vector3f phase_translation{
            (translation_velocities[0] + translation_velocities[2] +
             translation_velocities[4]) /
            3.0f};
        const Eigen::Vector3f short_rotation{
            0.5f * (rotation_velocities[3] + rotation_velocities[4])};
        const Eigen::Vector3f short_translation{
            0.5f * (translation_velocities[3] + translation_velocities[4])};
        const float phase_disagreement{
            0.5f * ((rotation_velocities[4] - rotation_velocities[2]).norm() +
                    (rotation_velocities[2] - rotation_velocities[0]).norm())};
        const float adjacent_disagreement{
            (rotation_velocities[4] - rotation_velocities[3]).norm()};
        const bool phase_is_more_consistent{
            phase_disagreement < adjacent_disagreement};
        const Eigen::Vector3f prior_rotation_vector{
            phase_is_more_consistent ? phase_rotation : short_rotation};
        const Eigen::Vector3f prior_translation{
            phase_is_more_consistent ? phase_translation : short_translation};
        const auto make_variation = [](const Eigen::Vector3f &rotation_vector,
                                       const Eigen::Vector3f &translation) {
          m3t::Transform3fA variation{m3t::Transform3fA::Identity()};
          variation.translate(translation);
          const float rotation_angle{rotation_vector.norm()};
          if (rotation_angle > std::numeric_limits<float>::min())
            variation.rotate(Eigen::AngleAxisf{
                rotation_angle, rotation_vector / rotation_angle});
          return variation;
        };
        predicted_pose =
            last_pose * make_variation(short_rotation, short_translation);
        phase_prior_pose =
            last_pose * make_variation(prior_rotation_vector,
                                       prior_translation);
        phase_prior_pose_valid = true;
      } else if (use_windowed_temporal_prior_ &&
          successful_pose_history.size() >= 3) {
        const size_t n_velocities{
            std::min<size_t>(5, successful_pose_history.size() - 1)};
        Eigen::Vector3f mean_rotation_vector{Eigen::Vector3f::Zero()};
        Eigen::Vector3f mean_translation{Eigen::Vector3f::Zero()};
        const size_t first_pose_index{
            successful_pose_history.size() - n_velocities - 1};
        for (size_t history_index = first_pose_index;
             history_index + 1 < successful_pose_history.size();
             ++history_index) {
          const m3t::Transform3fA variation{
              successful_pose_history[history_index].inverse() *
              successful_pose_history[history_index + 1]};
          const Eigen::AngleAxisf rotation{variation.rotation().matrix()};
          if (std::isfinite(rotation.angle()))
            mean_rotation_vector += rotation.angle() * rotation.axis();
          mean_translation += variation.translation();
        }
        mean_rotation_vector /= float(n_velocities);
        mean_translation /= float(n_velocities);
        const size_t history_size{successful_pose_history.size()};
        const m3t::Transform3fA previous_variation{
            successful_pose_history[history_size - 3].inverse() *
            successful_pose_history[history_size - 2]};
        const m3t::Transform3fA current_variation{
            successful_pose_history[history_size - 2].inverse() *
            successful_pose_history[history_size - 1]};
        const Eigen::AngleAxisf previous_rotation{
            previous_variation.rotation().matrix()};
        const Eigen::AngleAxisf current_rotation{
            current_variation.rotation().matrix()};
        const Eigen::Vector3f short_rotation_vector{
            0.5f * (previous_rotation.angle() * previous_rotation.axis() +
                    current_rotation.angle() * current_rotation.axis())};
        const Eigen::Vector3f short_translation{
            0.5f * (previous_variation.translation() +
                    current_variation.translation())};
        mean_rotation_vector =
            0.5f * (mean_rotation_vector + short_rotation_vector);
        mean_translation = 0.5f * (mean_translation + short_translation);
        m3t::Transform3fA mean_variation{m3t::Transform3fA::Identity()};
        mean_variation.translate(mean_translation);
        const float mean_rotation_angle{mean_rotation_vector.norm()};
        if (mean_rotation_angle > std::numeric_limits<float>::min())
          mean_variation.rotate(Eigen::AngleAxisf{
              mean_rotation_angle,
              mean_rotation_vector / mean_rotation_angle});
        predicted_pose = last_pose * mean_variation;
      } else if (use_directional_temporal_prior_ &&
                 smoothed_prediction_ready) {
        const m3t::Transform3fA previous_variation{
            previous_previous_pose.inverse() * previous_pose};
        const m3t::Transform3fA last_variation{
            previous_pose.inverse() * last_pose};
        const Eigen::AngleAxisf previous_rotation{
            previous_variation.rotation().matrix()};
        const Eigen::AngleAxisf last_rotation{
            last_variation.rotation().matrix()};
        const Eigen::Vector3f mean_rotation_vector{
            0.5f * (previous_rotation.angle() * previous_rotation.axis() +
                    last_rotation.angle() * last_rotation.axis())};
        m3t::Transform3fA mean_variation{m3t::Transform3fA::Identity()};
        mean_variation.translate(
            0.5f * (previous_variation.translation() +
                    last_variation.translation()));
        const float mean_rotation_angle{mean_rotation_vector.norm()};
        if (mean_rotation_angle > std::numeric_limits<float>::min())
          mean_variation.rotate(Eigen::AngleAxisf{
              mean_rotation_angle,
              mean_rotation_vector / mean_rotation_angle});
        predicted_pose = last_pose * mean_variation;
      }
      predicted_pose_for_frame = predicted_pose;
      predicted_pose_valid = true;
      if (use_constant_velocity_prediction_)
        body_ptr->set_body2world_pose(predicted_pose);
      if (use_directional_temporal_prior_ || use_windowed_temporal_prior_ ||
          use_phase_consistent_temporal_prior_) {
        tracker_ptr->optimizer_ptrs()[0]
            ->root_link_ptr()
            ->SetDirectionalTemporalPrior(
                phase_prior_pose_valid ? phase_prior_pose : predicted_pose);
      }
    } else if (use_directional_temporal_prior_ ||
               use_windowed_temporal_prior_ ||
               use_phase_consistent_temporal_prior_) {
      tracker_ptr->optimizer_ptrs()[0]
          ->root_link_ptr()
          ->ClearDirectionalTemporalPrior();
    }
    if (orb_tracker_ptr) {
      if ((use_constant_velocity_prediction_ ||
           use_directional_temporal_prior_) &&
          orb_prediction_ready) {
        m3t::Transform3fA orb_predicted_pose{
            orb_last_pose * orb_previous_pose.inverse() * orb_last_pose};
        if (use_directional_temporal_prior_ &&
            orb_smoothed_prediction_ready) {
          const m3t::Transform3fA previous_variation{
              orb_previous_previous_pose.inverse() * orb_previous_pose};
          const m3t::Transform3fA last_variation{
              orb_previous_pose.inverse() * orb_last_pose};
          const Eigen::AngleAxisf previous_rotation{
              previous_variation.rotation().matrix()};
          const Eigen::AngleAxisf last_rotation{
              last_variation.rotation().matrix()};
          const Eigen::Vector3f mean_rotation_vector{
              0.5f * (previous_rotation.angle() * previous_rotation.axis() +
                      last_rotation.angle() * last_rotation.axis())};
          m3t::Transform3fA mean_variation{m3t::Transform3fA::Identity()};
          mean_variation.translate(
              0.5f * (previous_variation.translation() +
                      last_variation.translation()));
          const float mean_rotation_angle{mean_rotation_vector.norm()};
          if (mean_rotation_angle > std::numeric_limits<float>::min())
            mean_variation.rotate(Eigen::AngleAxisf{
                mean_rotation_angle,
                mean_rotation_vector / mean_rotation_angle});
          orb_predicted_pose = orb_last_pose * mean_variation;
        }
        if (use_constant_velocity_prediction_)
          orb_body_ptr->set_body2world_pose(orb_predicted_pose);
        if (use_directional_temporal_prior_)
          orb_tracker_ptr->optimizer_ptrs()[0]
              ->root_link_ptr()
              ->SetDirectionalTemporalPrior(orb_predicted_pose);
      } else if (use_directional_temporal_prior_) {
        orb_tracker_ptr->optimizer_ptrs()[0]
            ->root_link_ptr()
            ->ClearDirectionalTemporalPrior();
      }
    }
    results[i].frame_index = i;
    if (predicted_pose_valid) {
      Result prediction_result;
      CalculatePoseResults(predicted_pose_for_frame, poses_gt_first_[i + 1],
                           &prediction_result);
      results[i].prediction_valid = true;
      results[i].prediction_translation_error =
          prediction_result.translation_error;
      results[i].prediction_rotation_error = prediction_result.rotation_error;
      results[i].prediction_success = prediction_result.tracking_success;
    }
    const auto begin_dual_cycle{std::chrono::high_resolution_clock::now()};
    ExecutionTimes orb_execution_times;
    std::future<void> orb_tracking_future;
    if (use_dual_model_texture_hypothesis_) {
      orb_tracking_future = std::async(std::launch::async, [&]() {
        ExecuteMeasuredTrackingCycle(orb_tracker_ptr, i,
                                     &orb_execution_times, false);
      });
    }
    ExecuteMeasuredTrackingCycle(
        tracker_ptr, i, &results[i].execution_times,
        !use_validated_prediction_recovery_ &&
            !use_dual_model_texture_hypothesis_);
    if (use_dual_model_texture_hypothesis_) orb_tracking_future.get();
    if (!use_dual_model_texture_hypothesis_ &&
        (collect_diagnostics_ || use_selective_recovery_))
      CollectDiagnostics(tracker_ptr, &results[i].fusion_diagnostics);

    if (use_dual_model_texture_hypothesis_) {
      FusionDiagnostics disk_diagnostics;
      FusionDiagnostics orb_diagnostics;
      CollectDiagnostics(tracker_ptr, &disk_diagnostics);
      CollectDiagnostics(orb_tracker_ptr, &orb_diagnostics);

      // Compare two fully optimized pose hypotheses using only evidence that
      // is observable online. Region agreement dominates the posterior;
      // cross-domain residual and reliability provide a small appearance
      // correction. Positive log odds select the DISK hypothesis.
      const float delta_region_residual{
          disk_diagnostics.region.residual_mean -
          orb_diagnostics.region.residual_mean};
      const float delta_model_texture_residual{
          disk_diagnostics.model_texture_residual_mean -
          orb_diagnostics.model_texture_residual_mean};
      const float delta_model_texture_reliability{
          disk_diagnostics.model_texture_reliability_weight -
          orb_diagnostics.model_texture_reliability_weight};
      const float disk_log_odds{
          0.17508205f - 7.8928036f * delta_region_residual -
          0.00471869f * delta_model_texture_residual +
          1.7693542f * delta_model_texture_reliability};
      const bool select_disk{disk_log_odds >= 0.0f};
      const m3t::Transform3fA disk_pose{body_ptr->body2world_pose()};
      const m3t::Transform3fA orb_pose{orb_body_ptr->body2world_pose()};
      Result disk_pose_result;
      Result orb_pose_result;
      CalculatePoseResults(disk_pose, poses_gt_first_[i + 1],
                           &disk_pose_result);
      CalculatePoseResults(orb_pose, poses_gt_first_[i + 1],
                           &orb_pose_result);
      const m3t::Transform3fA selected_pose{select_disk ? disk_pose : orb_pose};
      selected_pose_for_evaluation = selected_pose;
      results[i].fusion_diagnostics =
          select_disk ? disk_diagnostics : orb_diagnostics;
      results[i].fusion_diagnostics.dual_hypothesis_log_odds = disk_log_odds;
      results[i].fusion_diagnostics.dual_hypothesis_selected_disk =
          select_disk;
      results[i].fusion_diagnostics.dual_hypothesis_disk_success =
          disk_pose_result.tracking_success;
      results[i].fusion_diagnostics.dual_hypothesis_orb_success =
          orb_pose_result.tracking_success;


      const auto begin_results_time{
          std::chrono::high_resolution_clock::now()};
      tracker_ptr->CalculateResults(i);
      orb_tracker_ptr->CalculateResults(i);
      tracker_ptr->VisualizeResults(i);
      if (visualize_all_results_) tracker_ptr->UpdateViewers(i);
      const float results_time{ElapsedTime(begin_results_time)};
      results[i].execution_times.calculate_correspondences +=
          orb_execution_times.calculate_correspondences;
      results[i].execution_times.calculate_gradient_and_hessian +=
          orb_execution_times.calculate_gradient_and_hessian;
      results[i].execution_times.calculate_optimization +=
          orb_execution_times.calculate_optimization;
      results[i].execution_times.calculate_results = results_time;
      results[i].execution_times.complete_cycle =
          ElapsedTime(begin_dual_cycle);
    }

    if (use_validated_prediction_recovery_ && predicted_pose_valid) {
      const auto begin_validation_time{
          std::chrono::high_resolution_clock::now()};
      std::shared_ptr<m3t::TextureModality> texture_ptr;
      std::shared_ptr<m3t::RegionModality> region_ptr;
      for (const auto &modality_ptr : tracker_ptr->optimizer_ptrs()[0]
                                            ->root_link_ptr()
                                            ->modality_ptrs()) {
        if (!texture_ptr)
          texture_ptr =
              std::dynamic_pointer_cast<m3t::TextureModality>(modality_ptr);
        if (!region_ptr)
          region_ptr =
              std::dynamic_pointer_cast<m3t::RegionModality>(modality_ptr);
      }
      if (texture_ptr && region_ptr) {
        const m3t::Transform3fA optimized_pose{
            body_ptr->body2world_pose()};
        const float optimized_score{
            region_ptr->CalculateSilhouetteValidationScore(
                texture_ptr->silhouette_renderer_ptr())};
        body_ptr->set_body2world_pose(predicted_pose_for_frame);
        const float prediction_score{
            region_ptr->CalculateSilhouetteValidationScore(
                texture_ptr->silhouette_renderer_ptr())};
        if (prediction_score > optimized_score) {
          for (const auto &renderer_ptr :
               tracker_ptr->start_modality_renderer_ptrs())
            renderer_ptr->StartRendering();
          // The region histogram must remain frozen until validation is
          // complete. Only refresh the texture keyframe after accepting the
          // prediction; RegionModality::CalculateResults() will update its
          // histogram exactly once at the selected pose below.
          texture_ptr->StartModality(i, 0);
          results[i].fusion_diagnostics.selective_recovery = true;
        } else {
          body_ptr->set_body2world_pose(optimized_pose);
        }
      }
      const auto begin_results_time{
          std::chrono::high_resolution_clock::now()};
      tracker_ptr->CalculateResults(i);
      tracker_ptr->VisualizeResults(i);
      if (visualize_all_results_) tracker_ptr->UpdateViewers(i);
      results[i].execution_times.calculate_results +=
          ElapsedTime(begin_results_time);
      const float validation_time{ElapsedTime(begin_validation_time)};
      results[i].execution_times.complete_cycle += validation_time;
    }

    if (use_selective_recovery_ && predicted_pose_valid) {
      const auto &diagnostics{results[i].fusion_diagnostics};
      const bool weak_region{
          diagnostics.region.normalized_information_min_eigenvalue < 1.0f};
      const bool weak_texture{diagnostics.texture.n_gradient_terms < 3 ||
                              diagnostics.texture.residual_mean > 3.0f};
      if (weak_region && weak_texture) {
        const auto begin_recovery_time{
            std::chrono::high_resolution_clock::now()};
        body_ptr->set_body2world_pose(predicted_pose_for_frame);
        for (const auto &renderer_ptr :
             tracker_ptr->start_modality_renderer_ptrs())
          renderer_ptr->StartRendering();
        for (const auto &modality_ptr : tracker_ptr->optimizer_ptrs()[0]
                                            ->root_link_ptr()
                                            ->modality_ptrs())
          modality_ptr->StartModality(i, 0);
        const float recovery_time{ElapsedTime(begin_recovery_time)};
        results[i].execution_times.calculate_results += recovery_time;
        results[i].execution_times.complete_cycle += recovery_time;
        results[i].fusion_diagnostics.selective_recovery = true;
      }
    }

    // Calculate results for main body
    CalculatePoseResults(
        use_dual_model_texture_hypothesis_ ? selected_pose_for_evaluation
                                           : body_ptr->body2world_pose(),
        poses_gt_first_[i + 1], &results[i]);
    if (visualize_all_results_)
      VisualizeFrameResult(results[i], run_configuration.sequence_name + ": " +
                                           run_configuration.body_name);
    if (use_dual_model_texture_hypothesis_) {
      const auto reset_disk_state = [&]() {
        ResetBody(tracker_ptr, i + 1);
        last_pose = body_ptr->body2world_pose();
        prediction_ready = false;
        smoothed_prediction_ready = false;
        successful_pose_history.clear();
        successful_pose_history.push_back(last_pose);
      };
      const auto update_disk_state = [&]() {
        previous_previous_pose = previous_pose;
        previous_pose = last_pose;
        last_pose = body_ptr->body2world_pose();
        smoothed_prediction_ready = prediction_ready;
        prediction_ready = true;
        successful_pose_history.push_back(last_pose);
        if (successful_pose_history.size() > 6)
          successful_pose_history.pop_front();
      };
      const auto reset_orb_state = [&]() {
        ResetBody(orb_tracker_ptr, i + 1);
        orb_last_pose = orb_body_ptr->body2world_pose();
        orb_prediction_ready = false;
        orb_smoothed_prediction_ready = false;
      };
      const auto update_orb_state = [&]() {
        orb_previous_previous_pose = orb_previous_pose;
        orb_previous_pose = orb_last_pose;
        orb_last_pose = orb_body_ptr->body2world_pose();
        orb_smoothed_prediction_ready = orb_prediction_ready;
        orb_prediction_ready = true;
      };
      if (use_dual_hypothesis_independent_benchmark_reset_) {
        if (results[i]
                .fusion_diagnostics.dual_hypothesis_disk_success == 0.0f)
          reset_disk_state();
        else
          update_disk_state();
        if (results[i].fusion_diagnostics.dual_hypothesis_orb_success ==
            0.0f)
          reset_orb_state();
        else
          update_orb_state();
      } else if (results[i].tracking_success == 0.0f) {
        reset_disk_state();
        reset_orb_state();
      } else {
        update_disk_state();
        update_orb_state();
      }
    } else if (results[i].tracking_success == 0.0f) {
      ResetBody(tracker_ptr, i + 1);
      if (use_constant_velocity_prediction_ ||
          use_directional_temporal_prior_ || use_windowed_temporal_prior_ ||
          use_phase_consistent_temporal_prior_ ||
          use_selective_recovery_ || use_validated_prediction_recovery_) {
        last_pose = body_ptr->body2world_pose();
        prediction_ready = false;
        smoothed_prediction_ready = false;
        successful_pose_history.clear();
        successful_pose_history.push_back(last_pose);
      }
    } else if (use_constant_velocity_prediction_ ||
               use_directional_temporal_prior_ ||
               use_windowed_temporal_prior_ ||
               use_phase_consistent_temporal_prior_ ||
               use_selective_recovery_ ||
               use_validated_prediction_recovery_) {
      previous_previous_pose = previous_pose;
      previous_pose = last_pose;
      last_pose = body_ptr->body2world_pose();
      smoothed_prediction_ready = prediction_ready;
      prediction_ready = true;
      successful_pose_history.push_back(last_pose);
      if (successful_pose_history.size() > 6)
        successful_pose_history.pop_front();
    }

    // Calculate results for occluding body
    if (run_configuration.occlusions) {
      Result occlusion_result;
      CalculatePoseResults(body_ptr->body2world_pose(), poses_gt_second_[i + 1],
                           &occlusion_result);
      if (occlusion_result.tracking_success == 0.0f)
        ResetOcclusionBody(tracker_ptr, i + 1);
    }
  }
  CalculateAverageResult(results, average_result);
  if (collect_diagnostics_) {
    const std::string title{
        run_configuration.sequence_name + "_" +
        (run_configuration.occlusions ? "modeled_" : "") +
        run_configuration.body_name};
    SaveDiagnostics(results, title);
  }
}

void RBOTEvaluator::SetUpTracker(
    const RunConfiguration &run_configuration,
    const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
    std::shared_ptr<m3t::Tracker> *tracker_ptr,
    bool force_orb_model_texture) const {
  *tracker_ptr = std::make_shared<m3t::Tracker>("tracker");
  renderer_geometry_ptr->ClearBodies();

  // Init camera
  auto camera_ptr{std::make_shared<m3t::LoaderColorCamera>(
      "camera", dataset_directory_ / run_configuration.body_name / "frames",
      kRBOTIntrinsics, run_configuration.sequence_name, 0, 4)};
  camera_ptr->set_use_noise_adaptive_filtering(use_noise_adaptive_filtering_);
  camera_ptr->set_use_gaussian_noise_filter(use_gaussian_noise_filter_);
  camera_ptr->SetUp();

  // Init Viewer
  if (visualize_all_results_ && !force_orb_model_texture) {
    auto viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
        "viewer", camera_ptr, renderer_geometry_ptr)};
    viewer_ptr->SetUp();
    (*tracker_ptr)->AddViewer(viewer_ptr);
  }

  // Init bodies
  const std::string &body_name{run_configuration.body_name};
  auto body_ptr{std::make_shared<m3t::Body>(*body2body_ptr_map_.at(body_name))};
  renderer_geometry_ptr->AddBody(body_ptr);
  auto occlusion_body_ptr{
      std::make_shared<m3t::Body>(*body2body_ptr_map_.at(kOcclusionBodyName))};
  if (run_configuration.occlusions)
    renderer_geometry_ptr->AddBody(occlusion_body_ptr);

  // Init renderer
  auto silhouette_renderer_ptr{std::make_shared<m3t::FocusedSilhouetteRenderer>(
      "silhouette_renderer", renderer_geometry_ptr, camera_ptr)};
  silhouette_renderer_ptr->AddReferencedBody(body_ptr);
  if (run_configuration.occlusions)
    silhouette_renderer_ptr->AddReferencedBody(occlusion_body_ptr);
  silhouette_renderer_ptr->SetUp();

  auto depth_renderer_ptr{std::make_shared<m3t::FocusedBasicDepthRenderer>(
      "depth_renderer", renderer_geometry_ptr, camera_ptr)};
  depth_renderer_ptr->AddReferencedBody(body_ptr);
  depth_renderer_ptr->AddReferencedBody(occlusion_body_ptr);
  depth_renderer_setter_(depth_renderer_ptr);
  depth_renderer_ptr->SetUp();

  // Init link
  auto link_ptr{std::make_shared<m3t::Link>(body_name + "_link", body_ptr)};
  link_ptr->set_use_reliability_weighted_fusion(
      use_reliability_weighted_fusion_);

  // Init modalities
  if (use_region_modality_) {
    auto region_modality_ptr{std::make_shared<m3t::RegionModality>(
        body_name + "_region_modality", body_ptr, camera_ptr,
        body2model_ptr_map_.at(body_name))};
    region_modality_setter_(region_modality_ptr);
    region_modality_ptr->set_collect_diagnostics(
        collect_diagnostics_ || use_reliability_weighted_fusion_ ||
        use_selective_recovery_);
    region_modality_ptr->set_n_unoccluded_iterations(0);
    if (run_configuration.occlusions)
      region_modality_ptr->ModelOcclusions(depth_renderer_ptr);
    region_modality_ptr->SetUp();
    link_ptr->AddModality(region_modality_ptr);
  }

  if (use_texture_modality_) {
    auto texture_modality_ptr{std::make_shared<m3t::TextureModality>(
        body_name + "_texture_modality", body_ptr, camera_ptr,
        silhouette_renderer_ptr)};
    texture_modality_setter_(texture_modality_ptr);
    if (force_orb_model_texture) {
      texture_modality_ptr->set_use_model_texture_disk(false);
      texture_modality_ptr->set_use_adaptive_model_texture_disk(false);
      texture_modality_ptr->set_use_model_texture_disk_fusion(false);
    }
    texture_modality_ptr->set_collect_diagnostics(
        collect_diagnostics_ || use_reliability_weighted_fusion_ ||
        use_selective_recovery_);
    if (run_configuration.occlusions)
      texture_modality_ptr->ModelOcclusions(depth_renderer_ptr);
    texture_modality_ptr->SetUp();
    link_ptr->AddModality(texture_modality_ptr);
  }

  // Init optimizer
  link_ptr->SetUp();
  auto optimizer_ptr{
      std::make_shared<m3t::Optimizer>(body_name + "_optimizer", link_ptr)};
  optimizer_setter_(optimizer_ptr);
  optimizer_ptr->SetUp();
  (*tracker_ptr)->AddOptimizer(optimizer_ptr);

  if (run_configuration.occlusions) {
    // Init occlusion link
    auto occlusion_link_ptr{std::make_shared<m3t::Link>(
        kOcclusionBodyName + "_link", occlusion_body_ptr)};
    occlusion_link_ptr->set_use_reliability_weighted_fusion(
        use_reliability_weighted_fusion_);

    // Init occlusion modalities
    if (use_region_modality_) {
      auto occlusion_region_modality_ptr{std::make_shared<m3t::RegionModality>(
          kOcclusionBodyName + "_region_modality", occlusion_body_ptr,
          camera_ptr, body2model_ptr_map_.at(kOcclusionBodyName))};
      region_modality_setter_(occlusion_region_modality_ptr);
      occlusion_region_modality_ptr->set_collect_diagnostics(
          collect_diagnostics_);
      occlusion_region_modality_ptr->set_n_unoccluded_iterations(0);
      occlusion_region_modality_ptr->ModelOcclusions(depth_renderer_ptr);
      occlusion_region_modality_ptr->SetUp();
      occlusion_link_ptr->AddModality(occlusion_region_modality_ptr);
    }

    if (use_texture_modality_) {
      auto occlusion_texture_modality_ptr{
          std::make_shared<m3t::TextureModality>(
              kOcclusionBodyName + "_texture_modality", occlusion_body_ptr,
              camera_ptr, silhouette_renderer_ptr)};
      texture_modality_setter_(occlusion_texture_modality_ptr);
      if (force_orb_model_texture) {
        occlusion_texture_modality_ptr->set_use_model_texture_disk(false);
        occlusion_texture_modality_ptr
            ->set_use_adaptive_model_texture_disk(false);
        occlusion_texture_modality_ptr->set_use_model_texture_disk_fusion(
            false);
      }
      occlusion_texture_modality_ptr->set_collect_diagnostics(
          collect_diagnostics_);
      occlusion_texture_modality_ptr->ModelOcclusions(depth_renderer_ptr);
      occlusion_texture_modality_ptr->SetUp();
      occlusion_link_ptr->AddModality(occlusion_texture_modality_ptr);
    }

    // Init occlusion optimizer
    occlusion_link_ptr->SetUp();
    auto occlusion_optimizer_ptr{std::make_shared<m3t::Optimizer>(
        kOcclusionBodyName + "_optimizer", occlusion_link_ptr)};
    optimizer_setter_(occlusion_optimizer_ptr);
    occlusion_optimizer_ptr->SetUp();
    (*tracker_ptr)->AddOptimizer(occlusion_optimizer_ptr);
  }

  tracker_setter_(*tracker_ptr);
  (*tracker_ptr)->SetUp(false);
}

void RBOTEvaluator::ResetBody(const std::shared_ptr<m3t::Tracker> &tracker_ptr,
                              int i_frame) const {
  const auto &link_ptr{tracker_ptr->optimizer_ptrs()[0]->root_link_ptr()};
  link_ptr->body_ptr()->set_body2world_pose(poses_gt_first_[i_frame]);
  for (const auto &renderer_ptr : tracker_ptr->start_modality_renderer_ptrs())
    renderer_ptr->StartRendering();
  for (const auto &modality_ptr : link_ptr->modality_ptrs())
    modality_ptr->StartModality(0, 0);
}

void RBOTEvaluator::ResetOcclusionBody(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr, int i_frame) const {
  const auto &link_ptr{tracker_ptr->optimizer_ptrs()[1]->root_link_ptr()};
  link_ptr->body_ptr()->set_body2world_pose(poses_gt_second_[i_frame]);
  for (const auto &renderer_ptr : tracker_ptr->start_modality_renderer_ptrs())
    renderer_ptr->StartRendering();
  for (const auto &modality_ptr : link_ptr->modality_ptrs())
    modality_ptr->StartModality(0, 0);
}

void RBOTEvaluator::ExecuteMeasuredTrackingCycle(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr, int iteration,
    ExecutionTimes *execution_times, bool calculate_results) const {
  std::chrono::high_resolution_clock::time_point begin_time;

  // Update Cameras
  tracker_ptr->UpdateCameras(iteration);

  execution_times->calculate_correspondences = 0.0f;
  // Include image conditioning in the tracking budget, not hidden in I/O.
  for (const auto &camera : tracker_ptr->camera_ptrs()) {
    const auto loader = std::dynamic_pointer_cast<m3t::LoaderColorCamera>(camera);
    if (loader)
      execution_times->calculate_correspondences += loader->preprocessing_time_us();
  }
  execution_times->calculate_gradient_and_hessian = 0.0f;
  execution_times->calculate_optimization = 0.0f;
  for (int corr_iteration = 0;
       corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {
    // Calculate correspondences
    begin_time = std::chrono::high_resolution_clock::now();
    tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);
    execution_times->calculate_correspondences += ElapsedTime(begin_time);

    // Visualize correspondences
    int corr_save_idx =
        iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
    tracker_ptr->VisualizeCorrespondences(corr_save_idx);

    for (int update_iteration = 0;
         update_iteration < tracker_ptr->n_update_iterations();
         ++update_iteration) {
      // Calculate gradient and hessian
      begin_time = std::chrono::high_resolution_clock::now();
      tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration,
                                               update_iteration);
      execution_times->calculate_gradient_and_hessian +=
          ElapsedTime(begin_time);

      // Calculate optimization
      begin_time = std::chrono::high_resolution_clock::now();
      tracker_ptr->CalculateOptimization(iteration, corr_iteration,
                                         update_iteration);
      execution_times->calculate_optimization += ElapsedTime(begin_time);

      // Visualize optimization
      int update_save_idx =
          corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
      tracker_ptr->VisualizeOptimization(update_save_idx);
    }
  }

  execution_times->calculate_results = 0.0f;
  if (calculate_results) {
    // Calculate results
    begin_time = std::chrono::high_resolution_clock::now();
    tracker_ptr->CalculateResults(iteration);
    execution_times->calculate_results = ElapsedTime(begin_time);

    // Visualize results and update viewers
    tracker_ptr->VisualizeResults(iteration);
    if (visualize_all_results_) tracker_ptr->UpdateViewers(iteration);
  }

  execution_times->complete_cycle =
      execution_times->calculate_correspondences +
      execution_times->calculate_gradient_and_hessian +
      execution_times->calculate_optimization +
      execution_times->calculate_results;
}

void RBOTEvaluator::CollectDiagnostics(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr,
    FusionDiagnostics *diagnostics) const {
  for (const auto &camera : tracker_ptr->camera_ptrs()) {
    const auto loader = std::dynamic_pointer_cast<m3t::LoaderColorCamera>(camera);
    if (loader) {
      diagnostics->image_noise_sigma = loader->noise_standard_deviation();
      diagnostics->image_preprocessing_us = loader->preprocessing_time_us();
    }
  }
  const auto &optimizer_ptr{tracker_ptr->optimizer_ptrs()[0]};
  const auto &modalities{optimizer_ptr->root_link_ptr()->modality_ptrs()};
  for (const auto &modality_ptr : modalities) {
    if (auto region_ptr =
            std::dynamic_pointer_cast<m3t::RegionModality>(modality_ptr)) {
      diagnostics->region = CalculateModalityDiagnostics(
          modality_ptr, optimizer_ptr->tikhonov_parameter_rotation(),
          optimizer_ptr->tikhonov_parameter_translation());
      diagnostics->region.n_correspondences =
          int(region_ptr->n_correspondence_lines());
      diagnostics->region.n_gradient_terms =
          region_ptr->n_valid_gradient_lines();
      diagnostics->region.residual_mean =
          region_ptr->mean_absolute_standardized_residual();
      diagnostics->region.auxiliary_mean =
          region_ptr->mean_measured_variance();
      diagnostics->region_visibility_evidence = region_ptr->visibility_evidence();
      diagnostics->region_visibility_activation = region_ptr->visibility_activation();
    } else if (auto texture_ptr =
                   std::dynamic_pointer_cast<m3t::TextureModality>(
                       modality_ptr)) {
      diagnostics->texture = CalculateModalityDiagnostics(
          modality_ptr, optimizer_ptr->tikhonov_parameter_rotation(),
          optimizer_ptr->tikhonov_parameter_translation());
      diagnostics->texture.n_correspondences =
          int(texture_ptr->n_correspondence_points());
      diagnostics->texture.n_gradient_terms =
          texture_ptr->n_valid_gradient_points();
      diagnostics->texture.residual_mean =
          texture_ptr->mean_reprojection_error();
      diagnostics->texture.inlier_fraction = texture_ptr->inlier_fraction();
      diagnostics->model_texture_correspondences =
          int(texture_ptr->n_model_texture_correspondence_points());
      diagnostics->model_texture_residual_mean =
          texture_ptr->mean_model_texture_reprojection_error();
      diagnostics->model_photometric_points =
          texture_ptr->n_valid_model_photometric_points();
      diagnostics->model_photometric_residual_mean =
          texture_ptr->mean_model_photometric_residual();
      diagnostics->model_contour_points =
          texture_ptr->n_valid_model_contour_points();
      diagnostics->model_contour_residual_mean =
          texture_ptr->mean_model_contour_residual();
      diagnostics->model_ecc_points =
          texture_ptr->n_valid_model_ecc_points();
      diagnostics->model_ecc_residual_mean =
          texture_ptr->mean_model_ecc_residual();
      diagnostics->model_ecc_correlation =
          texture_ptr->model_ecc_correlation();
      diagnostics->model_hypothesis_confidence =
          texture_ptr->model_hypothesis_confidence();
      diagnostics->video_ecc_points =
          texture_ptr->n_valid_video_ecc_points();
      diagnostics->video_ecc_residual_mean =
          texture_ptr->mean_video_ecc_residual();
      diagnostics->video_ecc_correlation =
          texture_ptr->video_ecc_correlation();
    }
  }
  diagnostics->model_texture_reliability_weight =
      optimizer_ptr->root_link_ptr()
          ->last_model_texture_reliability_weight();
  diagnostics->model_photometric_reliability_weight =
      optimizer_ptr->root_link_ptr()
          ->last_model_photometric_reliability_weight();
  diagnostics->model_contour_reliability_weight =
      optimizer_ptr->root_link_ptr()
          ->last_model_contour_reliability_weight();
  diagnostics->model_ecc_reliability_weight =
      optimizer_ptr->root_link_ptr()->last_model_ecc_reliability_weight();
  diagnostics->model_hypothesis_reliability_weight =
      optimizer_ptr->root_link_ptr()
          ->last_model_hypothesis_reliability_weight();
  diagnostics->video_ecc_reliability_weight =
      optimizer_ptr->root_link_ptr()->last_video_ecc_reliability_weight();

  Eigen::Matrix<float, 6, 1> sqrt_tikhonov;
  sqrt_tikhonov.head<3>().setConstant(
      std::sqrt(optimizer_ptr->tikhonov_parameter_rotation()));
  sqrt_tikhonov.tail<3>().setConstant(
      std::sqrt(optimizer_ptr->tikhonov_parameter_translation()));
  const auto region_step{
      sqrt_tikhonov.cwiseProduct(diagnostics->region.regularized_step)};
  const auto texture_step{
      sqrt_tikhonov.cwiseProduct(diagnostics->texture.regularized_step)};
  const float denominator{region_step.norm() * texture_step.norm()};
  if (denominator > std::numeric_limits<float>::min()) {
    diagnostics->step_cosine =
        std::clamp(region_step.dot(texture_step) / denominator, -1.0f, 1.0f);
  }
  diagnostics->regularized_step_disagreement =
      (region_step - texture_step).norm();
}

RBOTEvaluator::ModalityDiagnostics
RBOTEvaluator::CalculateModalityDiagnostics(
    const std::shared_ptr<m3t::Modality> &modality_ptr,
    float tikhonov_parameter_rotation,
    float tikhonov_parameter_translation) {
  ModalityDiagnostics diagnostics;
  Eigen::Matrix<float, 6, 1> tikhonov;
  tikhonov.head<3>().setConstant(tikhonov_parameter_rotation);
  tikhonov.tail<3>().setConstant(tikhonov_parameter_translation);
  const auto inverse_sqrt_tikhonov{
      tikhonov.array().sqrt().inverse().matrix()};

  const Eigen::Matrix<float, 6, 6> information{
      -0.5f * (modality_ptr->hessian() + modality_ptr->hessian().transpose())};
  const Eigen::Matrix<float, 6, 6> normalized_information{
      inverse_sqrt_tikhonov.asDiagonal() * information *
      inverse_sqrt_tikhonov.asDiagonal()};
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eigen_solver{
      normalized_information};
  if (eigen_solver.info() == Eigen::Success) {
    diagnostics.normalized_information_min_eigenvalue =
        eigen_solver.eigenvalues()(0);
    diagnostics.normalized_information_max_eigenvalue =
        eigen_solver.eigenvalues()(5);
    diagnostics.normalized_information_trace =
        eigen_solver.eigenvalues().sum();
    diagnostics.normalized_information_condition =
        std::max(0.0f, diagnostics.normalized_information_max_eigenvalue) /
        std::max(1.0e-8f,
                 diagnostics.normalized_information_min_eigenvalue);
  }
  diagnostics.normalized_gradient_norm =
      inverse_sqrt_tikhonov.cwiseProduct(modality_ptr->gradient()).norm();

  Eigen::Matrix<float, 6, 6> regularized_information{information};
  regularized_information.diagonal() += tikhonov;
  diagnostics.regularized_step =
      regularized_information.ldlt().solve(modality_ptr->gradient());
  if (!diagnostics.regularized_step.allFinite())
    diagnostics.regularized_step.setZero();
  diagnostics.step_rotation_norm =
      diagnostics.regularized_step.head<3>().norm();
  diagnostics.step_translation_norm =
      diagnostics.regularized_step.tail<3>().norm();
  return diagnostics;
}

void RBOTEvaluator::CalculatePoseResults(
    const m3t::Transform3fA &body2world_pose,
    const m3t::Transform3fA &body2world_pose_gt, Result *result) const {
  result->translation_error = (body2world_pose.translation().matrix() -
                               body2world_pose_gt.translation().matrix())
                                  .norm();
  result->rotation_error =
      acos(((body2world_pose.rotation().matrix().transpose() *
             body2world_pose_gt.rotation().matrix())
                .trace() -
            1.0f) /
           2.0f);
  if (result->translation_error > translation_error_threshold_ ||
      result->rotation_error > rotation_error_threshold_)
    result->tracking_success = 0.0f;
  else
    result->tracking_success = 1.0f;
}

void RBOTEvaluator::CalculateAverageResult(const std::vector<Result> &results,
                                           Result *average_result) {
  average_result->rotation_error = 0.0f;
  average_result->translation_error = 0.0f;
  average_result->tracking_success = 0.0f;
  average_result->execution_times.calculate_correspondences = 0.0f;
  average_result->execution_times.calculate_gradient_and_hessian = 0.0f;
  average_result->execution_times.calculate_optimization = 0.0f;
  average_result->execution_times.calculate_results = 0.0f;
  average_result->execution_times.complete_cycle = 0.0f;

  for (auto &result : results) {
    average_result->rotation_error += result.rotation_error;
    average_result->translation_error += result.translation_error;
    average_result->tracking_success += result.tracking_success;
    average_result->execution_times.calculate_correspondences +=
        result.execution_times.calculate_correspondences;
    average_result->execution_times.calculate_gradient_and_hessian +=
        result.execution_times.calculate_gradient_and_hessian;
    average_result->execution_times.calculate_optimization +=
        result.execution_times.calculate_optimization;
    average_result->execution_times.calculate_results +=
        result.execution_times.calculate_results;
    average_result->execution_times.complete_cycle +=
        result.execution_times.complete_cycle;
  }

  float n = float(results.size());
  average_result->frame_index = 0;
  average_result->rotation_error /= n;
  average_result->translation_error /= n;
  average_result->tracking_success /= n;
  average_result->execution_times.calculate_correspondences /= n;
  average_result->execution_times.calculate_gradient_and_hessian /= n;
  average_result->execution_times.calculate_optimization /= n;
  average_result->execution_times.calculate_results /= n;
  average_result->execution_times.complete_cycle /= n;
}

void RBOTEvaluator::VisualizeFrameResult(const Result &result,
                                         const std::string &title) {
  std::cout << title << ": "
            << "frame " << result.frame_index << ": "
            << "execution_time = " << result.execution_times.complete_cycle
            << " us "
            << "rotation_error = " << result.rotation_error * 180.0f / m3t::kPi
            << ", "
            << "translation_error = " << result.translation_error << ", "
            << "tracking success = " << result.tracking_success << std::endl;
}

void RBOTEvaluator::VisualizeFinalResult(const Result &results,
                                         const std::string &title) {
  std::cout << std::string(80, '-') << std::endl;
  std::cout << title << ":" << std::endl;
  std::cout << "success rate = " << results.tracking_success << std::endl;
  std::cout << "execution times:" << std::endl;
  std::cout << "complete cycle = " << results.execution_times.complete_cycle
            << " us" << std::endl;
  std::cout << "calculate correspondences = "
            << results.execution_times.calculate_correspondences << " us"
            << std::endl;
  std::cout << "calculate gradient and hessian = "
            << results.execution_times.calculate_gradient_and_hessian << " us"
            << std::endl;
  std::cout << "calculate optimization = "
            << results.execution_times.calculate_optimization << " us"
            << std::endl;
  std::cout << "calculate results = "
            << results.execution_times.calculate_results << " us" << std::endl;
}

void RBOTEvaluator::SaveFinalResult(const Result &result,
                                    const std::string &title) const {
  std::ofstream ofs{save_directory_ / ("results_" + title + ".txt")};
  ofs << result.tracking_success << "," << result.execution_times.complete_cycle
      << "," << result.execution_times.calculate_correspondences << ","
      << result.execution_times.calculate_gradient_and_hessian << ","
      << result.execution_times.calculate_optimization << ","
      << result.execution_times.calculate_results << std::endl;
  ofs.flush();
  ofs.close();
}

void RBOTEvaluator::SaveDiagnostics(const std::vector<Result> &results,
                                    const std::string &title) const {
  std::ofstream ofs{diagnostics_directory_ /
                    ("diagnostics_" + title + ".csv")};
  ofs << "frame,translation_error_m,rotation_error_rad,tracking_success,"
         "prediction_valid,prediction_translation_error_m,"
         "prediction_rotation_error_rad,prediction_success,"
         "cycle_us,region_lines,region_gradient_lines,"
         "region_mean_standardized_residual,region_mean_variance,"
         "region_normalized_gradient_norm,region_information_trace,"
         "region_information_min_eigenvalue,"
         "region_information_max_eigenvalue,region_information_condition,"
         "region_step_rotation_rad,region_step_translation_m,"
         "texture_points,texture_gradient_points,"
         "texture_mean_reprojection_error_px,texture_inlier_fraction,"
         "texture_normalized_gradient_norm,texture_information_trace,"
         "texture_information_min_eigenvalue,"
         "texture_information_max_eigenvalue,texture_information_condition,"
         "texture_step_rotation_rad,texture_step_translation_m,"
         "modality_step_cosine,regularized_step_disagreement,"
         "model_texture_correspondences,model_texture_residual_mean_px,"
         "model_texture_reliability_weight,"
         "model_photometric_points,model_photometric_residual_mean,"
         "model_photometric_reliability_weight,"
         "model_contour_points,model_contour_residual_mean,"
         "model_contour_reliability_weight,"
         "model_ecc_points,model_ecc_residual_mean,model_ecc_correlation,"
         "model_ecc_reliability_weight,"
         "model_hypothesis_confidence,"
         "model_hypothesis_reliability_weight,"
         "video_ecc_points,video_ecc_residual_mean,video_ecc_correlation,"
         "video_ecc_reliability_weight,"
         "selective_recovery,dual_hypothesis_log_odds,"
         "dual_hypothesis_selected_disk,dual_hypothesis_disk_success,"
         "dual_hypothesis_orb_success,image_noise_sigma,image_preprocessing_us,"
         "region_visibility_evidence,region_visibility_activation\n";
  ofs << std::setprecision(9);
  for (const auto &result : results) {
    const auto &region{result.fusion_diagnostics.region};
    const auto &texture{result.fusion_diagnostics.texture};
    ofs << result.frame_index << ',' << result.translation_error << ','
        << result.rotation_error << ',' << result.tracking_success << ','
        << int(result.prediction_valid) << ','
        << result.prediction_translation_error << ','
        << result.prediction_rotation_error << ','
        << result.prediction_success << ','
        << result.execution_times.complete_cycle << ','
        << region.n_correspondences << ',' << region.n_gradient_terms << ','
        << region.residual_mean << ',' << region.auxiliary_mean << ','
        << region.normalized_gradient_norm << ','
        << region.normalized_information_trace << ','
        << region.normalized_information_min_eigenvalue << ','
        << region.normalized_information_max_eigenvalue << ','
        << region.normalized_information_condition << ','
        << region.step_rotation_norm << ',' << region.step_translation_norm
        << ',' << texture.n_correspondences << ','
        << texture.n_gradient_terms << ',' << texture.residual_mean << ','
        << texture.inlier_fraction << ','
        << texture.normalized_gradient_norm << ','
        << texture.normalized_information_trace << ','
        << texture.normalized_information_min_eigenvalue << ','
        << texture.normalized_information_max_eigenvalue << ','
        << texture.normalized_information_condition << ','
        << texture.step_rotation_norm << ',' << texture.step_translation_norm
        << ',' << result.fusion_diagnostics.step_cosine << ','
        << result.fusion_diagnostics.regularized_step_disagreement << ','
        << result.fusion_diagnostics.model_texture_correspondences << ','
        << result.fusion_diagnostics.model_texture_residual_mean << ','
        << result.fusion_diagnostics.model_texture_reliability_weight << ','
        << result.fusion_diagnostics.model_photometric_points << ','
        << result.fusion_diagnostics.model_photometric_residual_mean << ','
        << result.fusion_diagnostics.model_photometric_reliability_weight
        << ','
        << result.fusion_diagnostics.model_contour_points << ','
        << result.fusion_diagnostics.model_contour_residual_mean << ','
        << result.fusion_diagnostics.model_contour_reliability_weight << ','
        << result.fusion_diagnostics.model_ecc_points << ','
        << result.fusion_diagnostics.model_ecc_residual_mean << ','
        << result.fusion_diagnostics.model_ecc_correlation << ','
        << result.fusion_diagnostics.model_ecc_reliability_weight << ','
        << result.fusion_diagnostics.model_hypothesis_confidence << ','
        << result.fusion_diagnostics.model_hypothesis_reliability_weight
        << ','
        << result.fusion_diagnostics.video_ecc_points << ','
        << result.fusion_diagnostics.video_ecc_residual_mean << ','
        << result.fusion_diagnostics.video_ecc_correlation << ','
        << result.fusion_diagnostics.video_ecc_reliability_weight << ','
        << int(result.fusion_diagnostics.selective_recovery) << ','
        << result.fusion_diagnostics.dual_hypothesis_log_odds << ','
        << int(result.fusion_diagnostics.dual_hypothesis_selected_disk)
        << ',' << result.fusion_diagnostics.dual_hypothesis_disk_success
        << ',' << result.fusion_diagnostics.dual_hypothesis_orb_success
        << ',' << result.fusion_diagnostics.image_noise_sigma
        << ',' << result.fusion_diagnostics.image_preprocessing_us
        << ',' << result.fusion_diagnostics.region_visibility_evidence
        << ',' << result.fusion_diagnostics.region_visibility_activation
        << '\n';
  }
}

bool RBOTEvaluator::LoadBodies() {
  for (auto &body_name : body_names_) {
    if (!LoadSingleBody(body_name, dataset_directory_ / body_name))
      return false;
  }
  return LoadSingleBody(kOcclusionBodyName, dataset_directory_);
}

bool RBOTEvaluator::LoadSingleBody(const std::string &body_name,
                                   const std::filesystem::path &directory) {
  auto body_ptr{std::make_shared<m3t::Body>(
      body_name, directory / (body_name + ".obj"), 0.001f, true, false,
      m3t::Transform3fA::Identity())};
  if (!body_ptr->SetUp()) return false;
  body2body_ptr_map_.insert({body_name, std::move(body_ptr)});
  return true;
}

bool RBOTEvaluator::GenerateModels() {
  std::filesystem::path directory{external_directory_ / "models"};
  std::filesystem::create_directories(directory);
  for (auto &body_name : body_names_) {
    if (!GenerateSingleModel(body_name, directory)) return false;
  }
  return GenerateSingleModel(kOcclusionBodyName, directory);
}

bool RBOTEvaluator::GenerateSingleModel(
    const std::string &body_name, const std::filesystem::path &directory) {
  auto region_model_ptr{std::make_shared<m3t::RegionModel>(
      body_name + "_model", body2body_ptr_map_[body_name],
      directory / (body_name + "_model.bin"), 0.8f, 4, 200, 0.01f, 0.002f,
      false, 2000)};
  region_model_setter_(region_model_ptr);
  if (!region_model_ptr->SetUp()) return false;
  body2model_ptr_map_.insert({body_name, std::move(region_model_ptr)});
  return true;
}

bool RBOTEvaluator::ReadPosesRBOTDataset(
    const std::filesystem::path &path, std::vector<m3t::Transform3fA> *poses) {
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream " << path.string() << std::endl;
    return false;
  }

  poses->resize(kNFrames_ + 1);
  std::string parsed;
  std::getline(ifs, parsed);
  for (auto &pose : *poses) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        std::getline(ifs, parsed, '\t');
        pose.matrix()(i, j) = stof(parsed);
      }
    }
    std::getline(ifs, parsed, '\t');
    pose.matrix()(0, 3) = stof(parsed) * 0.001f;
    std::getline(ifs, parsed, '\t');
    pose.matrix()(1, 3) = stof(parsed) * 0.001f;
    std::getline(ifs, parsed);
    pose.matrix()(2, 3) = stof(parsed) * 0.001f;
  }
  return true;
}

float RBOTEvaluator::ElapsedTime(
    const std::chrono::high_resolution_clock::time_point &begin_time) {
  auto end_time{std::chrono::high_resolution_clock::now()};
  return float(std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     begin_time)
                   .count());
}
