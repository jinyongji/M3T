// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber and Mariam Elsayed,
// German Aerospace Center (DLR)

#include <m3t/texture_modality.h>
#include <m3t/loader_camera.h>

namespace m3t {

TextureModality::TextureModality(
    const std::string &name, const std::shared_ptr<Body> &body_ptr,
    const std::shared_ptr<ColorCamera> &color_camera_ptr,
    const std::shared_ptr<FocusedSilhouetteRenderer> &silhouette_renderer_ptr)
    : Modality{name, body_ptr},
      color_camera_ptr_{color_camera_ptr},
      silhouette_renderer_ptr_{silhouette_renderer_ptr} {}

TextureModality::TextureModality(
    const std::string &name, const std::filesystem::path &metafile_path,
    const std::shared_ptr<Body> &body_ptr,
    const std::shared_ptr<ColorCamera> &color_camera_ptr,
    const std::shared_ptr<FocusedSilhouetteRenderer> &silhouette_renderer_ptr)
    : Modality{name, metafile_path, body_ptr},
      color_camera_ptr_{color_camera_ptr},
      silhouette_renderer_ptr_{silhouette_renderer_ptr} {}

bool TextureModality::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;

  // Check if all required objects are set up
  if (!body_ptr_->set_up()) {
    std::cerr << "Body " << body_ptr_->name() << " was not set up" << std::endl;
    return false;
  }
  if (!color_camera_ptr_->set_up()) {
    std::cerr << "Color camera " << color_camera_ptr_->name()
              << " was not set up" << std::endl;
    return false;
  }
  if (!silhouette_renderer_ptr_->set_up()) {
    std::cerr << "Focused silhouette renderer "
              << silhouette_renderer_ptr_->name() << " was not set up"
              << std::endl;
    return false;
  }
  if (measure_occlusions_ && !depth_camera_ptr_->set_up()) {
    std::cerr << "Depth camera " << depth_camera_ptr_->name()
              << " was not set up" << std::endl;
    return false;
  }
  if (model_occlusions_ && !depth_renderer_ptr_->set_up()) {
    std::cerr << "Focused depth renderer " << depth_renderer_ptr_->name()
              << " was not set up" << std::endl;
    return false;
  }
  if ((use_model_texture_matching_ || use_model_photometric_tracking_ ||
       use_model_contour_tracking_ || use_model_ecc_tracking_ ||
       use_model_hypothesis_search_) &&
      !textured_renderer_ptr_->set_up()) {
    std::cerr << "Textured renderer " << textured_renderer_ptr_->name()
              << " was not set up" << std::endl;
    return false;
  }

  // Check if all required objects are correctly configured
  if (silhouette_renderer_ptr_->id_type() != IDType::BODY) {
    std::cerr << "Focused silhouette renderer "
              << silhouette_renderer_ptr_->name()
              << " does not use id_type BODY" << std::endl;
  }

  SetUpFeatureDetectorAndMatcher();
#ifdef USE_ONNXRUNTIME
  if (use_model_texture_disk_ || use_adaptive_model_texture_disk_ ||
      use_model_texture_disk_fusion_) {
    disk_feature_extractor_ = std::make_unique<DiskFeatureExtractor>();
    if (!disk_feature_extractor_->SetUp(model_texture_disk_model_path_,
                                        model_texture_disk_input_size_,
                                        true))
      return false;
  }
#else
  if (use_model_texture_disk_) {
    std::cerr << "DISK model texture matching requires a build configured "
                 "with USE_ONNXRUNTIME=ON"
              << std::endl;
    return false;
  }
#endif
  PrecalculateCameraVariables();
  PrecalculateRendererVariables();
  SetImshowVariables();

  set_up_ = true;
  return true;
}

void TextureModality::set_color_camera_ptr(
    const std::shared_ptr<ColorCamera> &color_camera_ptr) {
  color_camera_ptr_ = color_camera_ptr;
  set_up_ = false;
}

void TextureModality::set_silhouette_renderer_ptr(
    const std::shared_ptr<FocusedSilhouetteRenderer> &silhouette_renderer_ptr) {
  silhouette_renderer_ptr_ = silhouette_renderer_ptr;
  set_up_ = false;
}

void TextureModality::set_descriptor_type(
    TextureModality::DescriptorType descriptor_type) {
  descriptor_type_ = descriptor_type;
  set_up_ = false;
}

void TextureModality::set_focused_image_size(int focused_image_size) {
  focused_image_size_ = focused_image_size;
}

void TextureModality::set_use_optical_flow_tracking(
    bool use_optical_flow_tracking) {
  use_optical_flow_tracking_ = use_optical_flow_tracking;
}

void TextureModality::set_use_geometry_guided_matching(
    bool use_geometry_guided_matching) {
  use_geometry_guided_matching_ = use_geometry_guided_matching;
}

void TextureModality::set_use_model_texture_sift(
    bool use_model_texture_sift) {
  use_model_texture_sift_ = use_model_texture_sift;
  set_up_ = false;
}

void TextureModality::set_use_model_texture_akaze(
    bool use_model_texture_akaze) {
  use_model_texture_akaze_ = use_model_texture_akaze;
  set_up_ = false;
}

void TextureModality::set_use_model_texture_disk(
    bool use_model_texture_disk,
    const std::filesystem::path &model_path) {
  use_model_texture_disk_ = use_model_texture_disk;
  if (!model_path.empty()) model_texture_disk_model_path_ = model_path;
  set_up_ = false;
}

void TextureModality::set_use_adaptive_model_texture_disk(
    bool use_adaptive_model_texture_disk,
    const std::filesystem::path &model_path) {
  use_adaptive_model_texture_disk_ = use_adaptive_model_texture_disk;
  if (!model_path.empty()) model_texture_disk_model_path_ = model_path;
  set_up_ = false;
}

void TextureModality::set_use_model_texture_disk_fusion(
    bool use_model_texture_disk_fusion,
    const std::filesystem::path &model_path) {
  use_model_texture_disk_fusion_ = use_model_texture_disk_fusion;
  if (!model_path.empty()) model_texture_disk_model_path_ = model_path;
  set_up_ = false;
}

void TextureModality::set_use_video_ecc_tracking(bool use_video_ecc_tracking) {
  use_video_ecc_tracking_ = use_video_ecc_tracking;
  set_up_ = false;
}

void TextureModality::UseModelTextureMatching(
    const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr) {
  textured_renderer_ptr_ = textured_renderer_ptr;
  use_model_texture_matching_ = true;
  set_up_ = false;
}

void TextureModality::DoNotUseModelTextureMatching() {
  use_model_texture_matching_ = false;
  if (!use_model_photometric_tracking_ && !use_model_contour_tracking_ &&
      !use_model_ecc_tracking_ && !use_model_hypothesis_search_)
    textured_renderer_ptr_ = nullptr;
  set_up_ = false;
}

void TextureModality::UseModelPhotometricTracking(
    const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr) {
  textured_renderer_ptr_ = textured_renderer_ptr;
  use_model_photometric_tracking_ = true;
  set_up_ = false;
}

void TextureModality::DoNotUseModelPhotometricTracking() {
  use_model_photometric_tracking_ = false;
  if (!use_model_texture_matching_ && !use_model_contour_tracking_ &&
      !use_model_ecc_tracking_ && !use_model_hypothesis_search_)
    textured_renderer_ptr_ = nullptr;
  set_up_ = false;
}

void TextureModality::UseModelContourTracking(
    const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr) {
  textured_renderer_ptr_ = textured_renderer_ptr;
  use_model_contour_tracking_ = true;
  set_up_ = false;
}

void TextureModality::DoNotUseModelContourTracking() {
  use_model_contour_tracking_ = false;
  if (!use_model_texture_matching_ && !use_model_photometric_tracking_ &&
      !use_model_ecc_tracking_ && !use_model_hypothesis_search_)
    textured_renderer_ptr_ = nullptr;
  set_up_ = false;
}

void TextureModality::UseModelEccTracking(
    const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr) {
  textured_renderer_ptr_ = textured_renderer_ptr;
  use_model_ecc_tracking_ = true;
  set_up_ = false;
}

void TextureModality::DoNotUseModelEccTracking() {
  use_model_ecc_tracking_ = false;
  if (!use_model_texture_matching_ && !use_model_photometric_tracking_ &&
      !use_model_contour_tracking_ && !use_model_hypothesis_search_)
    textured_renderer_ptr_ = nullptr;
  set_up_ = false;
}

void TextureModality::UseModelHypothesisSearch(
    const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr) {
  textured_renderer_ptr_ = textured_renderer_ptr;
  use_model_hypothesis_search_ = true;
  set_up_ = false;
}

void TextureModality::DoNotUseModelHypothesisSearch() {
  use_model_hypothesis_search_ = false;
  if (!use_model_texture_matching_ && !use_model_photometric_tracking_ &&
      !use_model_contour_tracking_ && !use_model_ecc_tracking_)
    textured_renderer_ptr_ = nullptr;
  set_up_ = false;
}

void TextureModality::set_model_texture_debug_directory(
    const std::filesystem::path &debug_directory) {
  model_texture_debug_directory_ = debug_directory;
  model_texture_debug_images_saved_ = false;
}

void TextureModality::set_model_texture_disk_ransac_threshold(
    float threshold) {
  model_texture_disk_ransac_threshold_ = std::max(0.1f, threshold);
}

void TextureModality::set_model_texture_disk_input_size(int input_size) {
  model_texture_disk_input_size_ = std::max(16, input_size);
  set_up_ = false;
}

void TextureModality::set_use_adaptive_model_texture_geometry(
    bool use_adaptive_model_texture_geometry) {
  use_adaptive_model_texture_geometry_ =
      use_adaptive_model_texture_geometry;
}

void TextureModality::set_use_model_texture_pnp_geometry(
    bool use_model_texture_pnp_geometry) {
  use_model_texture_pnp_geometry_ = use_model_texture_pnp_geometry;
}

void TextureModality::set_tukey_norm_constant(float tukey_norm_constant) {
  tukey_norm_constant_ = tukey_norm_constant;
}

void TextureModality::set_descriptor_distance_threshold(
    float descriptor_distance_threshold) {
  descriptor_distance_threshold_ = descriptor_distance_threshold;
}

void TextureModality::set_standard_deviations(
    const std::vector<float> &standard_deviations) {
  standard_deviations_ = standard_deviations;
}

void TextureModality::set_max_keyframe_rotation_difference(
    float max_keyframe_rotation_difference) {
  max_keyframe_rotation_difference_ = max_keyframe_rotation_difference;
}

void TextureModality::set_max_keyframe_age(int max_keyframe_age) {
  max_keyframe_age_ = max_keyframe_age;
}

void TextureModality::set_n_keyframes(int n_keyframes) {
  n_keyframes_ = n_keyframes;
}

void TextureModality::set_orb_n_features(int orb_n_features) {
  orb_n_features_ = orb_n_features;
  set_up_ = false;
}

void TextureModality::set_orb_scale_factor(float orb_scale_factor) {
  orb_scale_factor_ = orb_scale_factor;
  set_up_ = false;
}

void TextureModality::set_orb_n_levels(int orb_n_levels) {
  orb_n_levels_ = orb_n_levels;
  set_up_ = false;
}

void TextureModality::set_brisk_threshold(int brisk_threshold) {
  brisk_threshold_ = brisk_threshold;
  set_up_ = false;
}

void TextureModality::set_brisk_octave(int brisk_octave) {
  brisk_octave_ = brisk_octave;
  set_up_ = false;
}

void TextureModality::set_brisk_pattern_scale(float brisk_pattern_scale) {
  brisk_pattern_scale_ = brisk_pattern_scale;
  set_up_ = false;
}

void TextureModality::set_daisy_radius(float daisy_radius) {
  daisy_radius_ = daisy_radius;
  set_up_ = false;
}

void TextureModality::set_daisy_q_radius(int daisy_q_radius) {
  daisy_q_radius_ = daisy_q_radius;
  set_up_ = false;
}

void TextureModality::set_daisy_q_theta(int daisy_q_theta) {
  daisy_q_theta_ = daisy_q_theta;
  set_up_ = false;
}

void TextureModality::set_daisy_q_hist(int daisy_q_hist) {
  daisy_q_hist_ = daisy_q_hist;
  set_up_ = false;
}

void TextureModality::set_freak_orientation_normalized(
    bool freak_orientation_normalized) {
  freak_orientation_normalized_ = freak_orientation_normalized;
  set_up_ = false;
}

void TextureModality::set_freak_scale_normalized(bool freak_scale_normalized) {
  freak_scale_normalized_ = freak_scale_normalized;
  set_up_ = false;
}

void TextureModality::set_freak_pattern_scale(float freak_pattern_scale) {
  freak_pattern_scale_ = freak_pattern_scale;
  set_up_ = false;
}

void TextureModality::set_freak_n_octaves(int freak_n_octaves) {
  freak_n_octaves_ = freak_n_octaves;
  set_up_ = false;
}

void TextureModality::set_sift_n_features(int sift_n_features) {
  sift_n_features_ = sift_n_features;
  set_up_ = false;
}

void TextureModality::set_sift_n_octave_layers(int sift_n_octave_layers) {
  sift_n_octave_layers_ = sift_n_octave_layers;
  set_up_ = false;
}

void TextureModality::set_sift_contrast_threshold(
    double sift_contrast_threshold) {
  sift_contrast_threshold_ = sift_contrast_threshold;
  set_up_ = false;
}

void TextureModality::set_sift_edge_threshold(double sift_edge_threshold) {
  sift_edge_threshold_ = sift_edge_threshold;
  set_up_ = false;
}

void TextureModality::set_sift_sigma(double sift_sigma) {
  sift_sigma_ = sift_sigma;
  set_up_ = false;
}

void TextureModality::MeasureOcclusions(
    const std::shared_ptr<DepthCamera> &depth_camera_ptr) {
  depth_camera_ptr_ = depth_camera_ptr;
  measure_occlusions_ = true;
  set_up_ = false;
}

void TextureModality::DoNotMeasureOcclusions() {
  measure_occlusions_ = false;
  set_up_ = false;
}

void TextureModality::ModelOcclusions(
    const std::shared_ptr<FocusedDepthRenderer> &depth_renderer_ptr) {
  depth_renderer_ptr_ = depth_renderer_ptr;
  model_occlusions_ = true;
  set_up_ = false;
}

void TextureModality::DoNotModelOcclusions() {
  depth_renderer_ptr_ = nullptr;
  model_occlusions_ = false;
  set_up_ = false;
}

void TextureModality::set_measured_occlusion_radius(
    float measured_occlusion_radius) {
  measured_occlusion_radius_ = measured_occlusion_radius;
}

void TextureModality::set_measured_occlusion_threshold(
    float measured_occlusion_threshold) {
  measured_occlusion_threshold_ = measured_occlusion_threshold;
}

void TextureModality::set_modeled_occlusion_radius(
    float modeled_occlusion_radius) {
  modeled_occlusion_radius_ = modeled_occlusion_radius;
}

void TextureModality::set_modeled_occlusion_threshold(
    float modeled_occlusion_threshold) {
  modeled_occlusion_threshold_ = modeled_occlusion_threshold;
}

void TextureModality::set_visualize_correspondences_correspondence(
    bool visualize_correspondences_correspondence) {
  visualize_correspondences_correspondence_ =
      visualize_correspondences_correspondence;
  SetImshowVariables();
}

void TextureModality::set_visualize_corresopndences_optimization(
    bool visualize_correspondence_optimization) {
  visualize_correspondences_optimization_ =
      visualize_correspondence_optimization;
  SetImshowVariables();
}

void TextureModality::set_visualize_points_result(
    bool visualize_points_result) {
  visualize_points_result_ = visualize_points_result;
  SetImshowVariables();
}

void TextureModality::set_visualize_points_depth_image_result(
    bool visualize_points_depth_image_result) {
  visualize_points_depth_image_result_ = visualize_points_depth_image_result;
  SetImshowVariables();
}

void TextureModality ::set_visualize_points_silhouette_rendering_result(
    bool visualize_points_silhouete_rendering_result) {
  visualize_points_silhouette_rendering_result_ =
      visualize_points_silhouete_rendering_result;
  SetImshowVariables();
}

void TextureModality ::set_visualize_points_depth_rendering_result(
    bool visualize_points_depth_rendering_result) {
  visualize_points_depth_rendering_result_ =
      visualize_points_depth_rendering_result;
  SetImshowVariables();
}

void TextureModality::set_visualization_min_depth(
    float visualization_min_depth) {
  visualization_min_depth_ = visualization_min_depth;
}

void TextureModality::set_visualization_max_depth(
    float visualization_max_depth) {
  visualization_max_depth_ = visualization_max_depth;
}

void TextureModality::set_collect_diagnostics(bool collect_diagnostics) {
  collect_diagnostics_ = collect_diagnostics;
}

bool TextureModality::StartModality(int iteration, int corr_iteration) {
  if (!IsSetup()) return false;

  PrecalculatePoseVariables();
  DetectAndComputeCorrKeypoints();
  ComputeKeyframeData();
  return true;
}

bool TextureModality::CalculateCorrespondences(int iteration,
                                               int corr_iteration) {
  if (!IsSetup()) return false;

  PrecalculatePoseVariables();
  PrecalculateIterationDependentVariables(corr_iteration);

  // Calculate matches and reconstruct data points
  if (corr_iteration == 0) {
    // Match descriptors
    DetectAndComputeCorrKeypoints();
    if (use_optical_flow_tracking_) {
      ComputeOpticalFlowCorrespondences();
    } else if (use_geometry_guided_matching_) {
      ComputeGeometryGuidedCorrespondences();
    } else {
      std::vector<std::vector<std::vector<cv::DMatch>>> knn_matches_keyframes;
#ifdef USE_CUDA
      if (descriptor_type_ == DescriptorType::ORB_CUDA) {
        for (const auto &descriptors_keyframe_cuda :
             descriptors_keyframes_cuda_) {
          std::vector<std::vector<cv::DMatch>> knn_matches;
          if (!descriptors_keyframe_cuda.empty() && !descriptors_cuda_.empty())
            descriptor_matcher_cuda_->knnMatch(
                descriptors_keyframe_cuda, descriptors_cuda_, knn_matches, 2);
          knn_matches_keyframes.push_back(std::move(knn_matches));
        }
      } else
#endif
      {
        for (const auto &descriptors_keyframe : descriptors_keyframes_) {
          std::vector<std::vector<cv::DMatch>> knn_matches;
          if (!descriptors_keyframe.empty() && !descriptors_.empty())
            descriptor_matcher_->knnMatch(descriptors_keyframe, descriptors_,
                                          knn_matches, 2);
          knn_matches_keyframes.push_back(std::move(knn_matches));
        }
      }

      // Compute data points
      data_points_.clear();
      for (size_t i = 0; i < knn_matches_keyframes.size(); ++i) {
        const auto &knn_matches{knn_matches_keyframes[i]};
        const auto &points_keyframe{points_keyframes_[i]};
        for (const auto &knn_match : knn_matches) {
          if (knn_match.size() < 2) continue;
          if (knn_match[0].distance / knn_match[1].distance >=
              descriptor_distance_threshold_)
            continue;

          DataPoint data_point;
          data_point.correspondence_center =
              Eigen::Vector2f{keypoints_[knn_match[0].trainIdx].pt.x,
                              keypoints_[knn_match[0].trainIdx].pt.y};
          data_point.center_f_body = points_keyframe[knn_match[0].queryIdx];
          data_points_.push_back(std::move(data_point));
        }
      }
    }
    if ((use_model_texture_matching_ || use_model_photometric_tracking_ ||
         use_model_contour_tracking_ || use_model_ecc_tracking_ ||
         use_model_hypothesis_search_) &&
        body_ptr_->has_texture() && textured_renderer_ptr_->StartRendering() &&
        textured_renderer_ptr_->IsBodyVisible(body_ptr_->name()) &&
        textured_renderer_ptr_->FetchColorImage() &&
        textured_renderer_ptr_->FetchDepthImage()) {
      if (use_model_texture_matching_) ComputeModelTextureCorrespondences();
      if (use_model_photometric_tracking_) ComputeModelPhotometricData();
      if (use_model_contour_tracking_) ComputeModelContourData();
      if (use_model_ecc_tracking_) ComputeModelEccData();
      if (use_model_hypothesis_search_) ComputeModelHypothesis();
    }
    if (use_video_ecc_tracking_) ComputeVideoEccCorrespondences();
  }

  // Compute iteration dependent data points
  for (auto &data_point : data_points_) {
    data_point.center_f_camera = body2camera_pose_ * data_point.center_f_body;
    const auto &center_f_camera{data_point.center_f_camera};
    data_point.center(0) = center_f_camera(0) * fu_ / center_f_camera(2) + ppu_;
    data_point.center(1) = center_f_camera(1) * fv_ / center_f_camera(2) + ppv_;
  }
  return true;
}

bool TextureModality::VisualizeCorrespondences(int save_idx) {
  if (!IsSetup()) return false;

  if (visualize_correspondences_correspondence_)
    VisualizeCorrespondences("correspondences_correspondence", save_idx);
  return true;
}

bool TextureModality::CalculateGradientAndHessian(int iteration,
                                                  int corr_iteration,
                                                  int opt_iteration) {
  if (!IsSetup()) return false;

  PrecalculatePoseVariables();
  model_texture_statistics_available_ = false;
  adaptive_model_texture_disk_selected_ = false;
  adaptive_model_texture_disk_evidence_ = 0;
  gradient_.setZero();
  hessian_.setZero();
  n_valid_gradient_points_ = 0;
  mean_reprojection_error_ = 0.0f;
  inlier_fraction_ = 0.0f;
  n_valid_model_texture_points_ = 0;
  mean_model_texture_reprojection_error_ = 0.0f;
  model_texture_inlier_fraction_ = 0.0f;
  model_texture_gradient_.setZero();
  model_texture_hessian_.setZero();
  n_valid_model_texture_disk_points_ = 0;
  mean_model_texture_disk_reprojection_error_ = 0.0f;
  model_texture_disk_gradient_.setZero();
  model_texture_disk_hessian_.setZero();
  n_valid_model_photometric_points_ = 0;
  mean_model_photometric_residual_ = 0.0f;
  model_photometric_gradient_.setZero();
  model_photometric_hessian_.setZero();
  n_valid_model_contour_points_ = 0;
  mean_model_contour_residual_ = 0.0f;
  model_contour_gradient_.setZero();
  model_contour_hessian_.setZero();
  n_valid_model_ecc_points_ = 0;
  mean_model_ecc_residual_ = 0.0f;
  model_ecc_gradient_.setZero();
  model_ecc_hessian_.setZero();
  model_hypothesis_gradient_.setZero();
  model_hypothesis_hessian_.setZero();
  n_valid_video_ecc_points_ = 0;
  mean_video_ecc_residual_ = 0.0f;
  video_ecc_gradient_.setZero();
  video_ecc_hessian_.setZero();
  if (model_hypothesis_confidence_ > 0.0f) {
    const Transform3fA current2target_pose{
        body_ptr_->body2world_pose().inverse() *
        model_hypothesis_target_pose_};
    const Eigen::AngleAxisf rotation_error{
        current2target_pose.rotation().matrix()};
    if (std::isfinite(rotation_error.angle())) {
      model_hypothesis_gradient_.head<3>() =
          1000.0f * rotation_error.angle() * rotation_error.axis();
      model_hypothesis_hessian_.topLeftCorner<3, 3>() =
          -1000.0f * Eigen::Matrix3f::Identity();
    }
  }

  // iterate over points
  for (auto &data_point : data_points_) {
    // Calculate center in camera frame and image
    data_point.center_f_camera = body2camera_pose_ * data_point.center_f_body;
    const auto &center_f_camera{data_point.center_f_camera};
    const auto &center_f_body{data_point.center_f_body};
    data_point.center(0) = center_f_camera(0) * fu_ / center_f_camera(2) + ppu_;
    data_point.center(1) = center_f_camera(1) * fv_ / center_f_camera(2) + ppv_;
    float x = center_f_camera(0);
    float y = center_f_camera(1);
    float z = center_f_camera(2);
    float z2 = z * z;

    // Calculate error
    Eigen::Vector2f diff{data_point.center - data_point.correspondence_center};
    float squared_error = diff.squaredNorm();
    float error = sqrtf(squared_error);

    if (data_point.from_model_texture) {
      ++n_valid_model_texture_points_;
      mean_model_texture_reprojection_error_ += error;
      if (error <= tukey_norm_constant_)
        model_texture_inlier_fraction_ += 1.0f;
      if (data_point.from_model_texture_disk) {
        ++n_valid_model_texture_disk_points_;
        mean_model_texture_disk_reprojection_error_ += error;
      }
    } else if (collect_diagnostics_) {
        ++n_valid_gradient_points_;
        mean_reprojection_error_ += error;
        if (error <= tukey_norm_constant_) inlier_fraction_ += 1.0f;
    }

    // Calculate weight with Tukey norm
    float weight = 1.0f / variance_;
    if (error > std::numeric_limits<float>::min())
      weight = (TukeyNorm(error) / squared_error) / variance_;

    // Calculate derivatives
    Eigen::Matrix<float, 2, 3> dx_dX;
    dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z, -y * fv_ / z2;
    Eigen::Matrix<float, 2, 3> dx_dtranslation{dx_dX * body2camera_rotation_};
    Eigen::Matrix<float, 2, 6> dx_dtheta;
    dx_dtheta << -dx_dtranslation * Vector2Skewsymmetric(center_f_body),
        dx_dtranslation;

    // Calculate gradient and hessian
    const Eigen::Matrix<float, 6, 1> gradient_summand{
        -(weight * diff.transpose() * dx_dtheta).transpose()};
    const Eigen::Matrix<float, 6, 6> hessian_summand{
        -(weight * dx_dtheta.transpose()) * dx_dtheta};
    if (data_point.from_model_texture) {
      model_texture_gradient_ += gradient_summand;
      model_texture_hessian_ += hessian_summand;
      if (data_point.from_model_texture_disk) {
        model_texture_disk_gradient_ += gradient_summand;
        model_texture_disk_hessian_ += hessian_summand;
      }
    } else {
      gradient_ += gradient_summand;
      hessian_.triangularView<Eigen::Lower>() += hessian_summand;
    }
  }
  hessian_ = hessian_.selfadjointView<Eigen::Lower>();
  model_texture_hessian_ =
      0.5f * (model_texture_hessian_ + model_texture_hessian_.transpose());
  model_texture_disk_hessian_ =
      0.5f * (model_texture_disk_hessian_ +
              model_texture_disk_hessian_.transpose());

  // Calculate the direct CAD photometric term. The rendered surface anchors
  // remain fixed for the frame while their image projection follows every
  // pose update.
  for (const auto &data_point : model_photometric_data_points_) {
    const Eigen::Vector3f center_f_camera{
        body2camera_pose_ * data_point.center_f_body};
    const float x{center_f_camera(0)};
    const float y{center_f_camera(1)};
    const float z{center_f_camera(2)};
    if (z <= std::numeric_limits<float>::min()) continue;
    const float u{x * fu_ / z + ppu_};
    const float v{y * fv_ / z + ppv_};
    if (u < 1.0f || u >= float(image_width_minus_1_) || v < 1.0f ||
        v >= float(image_height_minus_1_))
      continue;

    const int u0{int(u)};
    const int v0{int(v)};
    const float du{u - float(u0)};
    const float dv{v - float(v0)};
    const auto bilinear_sample = [u0, v0, du, dv](const cv::Mat &image) {
      const float top{(1.0f - du) * image.at<float>(v0, u0) +
                      du * image.at<float>(v0, u0 + 1)};
      const float bottom{(1.0f - du) * image.at<float>(v0 + 1, u0) +
                         du * image.at<float>(v0 + 1, u0 + 1)};
      return (1.0f - dv) * top + dv * bottom;
    };
    const float measured_intensity{
        bilinear_sample(model_photometric_gray_image_)};
    const float residual{
        measured_intensity -
        (model_photometric_scale_ * data_point.model_intensity +
         model_photometric_offset_)};
    const float absolute_residual{std::abs(residual)};
    ++n_valid_model_photometric_points_;
    mean_model_photometric_residual_ += absolute_residual;

    if (absolute_residual >= model_photometric_tukey_constant_) continue;
    const float normalized_residual{
        residual / model_photometric_tukey_constant_};
    const float robust_factor{
        1.0f - normalized_residual * normalized_residual};
    const float weight{robust_factor * robust_factor /
                       model_photometric_variance_};
    const Eigen::RowVector2f image_gradient{
        bilinear_sample(model_photometric_gradient_x_),
        bilinear_sample(model_photometric_gradient_y_)};
    const float z2{z * z};
    Eigen::Matrix<float, 2, 3> dx_dX;
    dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z,
        -y * fv_ / z2;
    const Eigen::Matrix<float, 2, 3> dx_dtranslation{
        dx_dX * body2camera_rotation_};
    Eigen::Matrix<float, 2, 6> dx_dtheta;
    dx_dtheta <<
        -dx_dtranslation * Vector2Skewsymmetric(data_point.center_f_body),
        dx_dtranslation;
    const Eigen::Matrix<float, 1, 6> jacobian{image_gradient * dx_dtheta};
    model_photometric_gradient_ -= weight * residual * jacobian.transpose();
    model_photometric_hessian_ -= weight * jacobian.transpose() * jacobian;
  }
  if (n_valid_model_photometric_points_ > 0)
    mean_model_photometric_residual_ /=
        float(n_valid_model_photometric_points_);

  for (const auto &data_point : model_contour_data_points_) {
    const Eigen::Vector3f center_f_camera{
        body2camera_pose_ * data_point.center_f_body};
    const float x{center_f_camera(0)};
    const float y{center_f_camera(1)};
    const float z{center_f_camera(2)};
    if (z <= std::numeric_limits<float>::min()) continue;
    const Eigen::Vector2f center{x * fu_ / z + ppu_, y * fv_ / z + ppv_};
    const float residual{
        data_point.normal.dot(center - data_point.correspondence_center)};
    const float absolute_residual{std::abs(residual)};
    ++n_valid_model_contour_points_;
    mean_model_contour_residual_ += absolute_residual;
    if (absolute_residual >= model_contour_tukey_constant_) continue;
    const float normalized_residual{residual /
                                    model_contour_tukey_constant_};
    const float robust_factor{
        1.0f - normalized_residual * normalized_residual};
    const float weight{robust_factor * robust_factor /
                       model_contour_variance_};
    const float z2{z * z};
    Eigen::Matrix<float, 2, 3> dx_dX;
    dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z,
        -y * fv_ / z2;
    const Eigen::Matrix<float, 2, 3> dx_dtranslation{
        dx_dX * body2camera_rotation_};
    Eigen::Matrix<float, 2, 6> dx_dtheta;
    dx_dtheta <<
        -dx_dtranslation * Vector2Skewsymmetric(data_point.center_f_body),
        dx_dtranslation;
    const Eigen::Matrix<float, 1, 6> jacobian{
        data_point.normal.transpose() * dx_dtheta};
    model_contour_gradient_ -= weight * residual * jacobian.transpose();
    model_contour_hessian_ -= weight * jacobian.transpose() * jacobian;
  }
  if (n_valid_model_contour_points_ > 0)
    mean_model_contour_residual_ /= float(n_valid_model_contour_points_);

  for (const auto &data_point : model_ecc_data_points_) {
    const Eigen::Vector3f center_f_camera{
        body2camera_pose_ * data_point.center_f_body};
    const float x{center_f_camera(0)};
    const float y{center_f_camera(1)};
    const float z{center_f_camera(2)};
    if (z <= std::numeric_limits<float>::min()) continue;
    const Eigen::Vector2f center{x * fu_ / z + ppu_, y * fv_ / z + ppv_};
    const Eigen::Vector2f residual_vector{
        center - data_point.correspondence_center};
    const float residual{residual_vector.norm()};
    ++n_valid_model_ecc_points_;
    mean_model_ecc_residual_ += residual;
    if (residual >= model_ecc_tukey_constant_) continue;
    const float normalized_residual{residual / model_ecc_tukey_constant_};
    const float robust_factor{
        1.0f - normalized_residual * normalized_residual};
    const float weight{robust_factor * robust_factor / model_ecc_variance_};
    const float z2{z * z};
    Eigen::Matrix<float, 2, 3> dx_dX;
    dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z,
        -y * fv_ / z2;
    const Eigen::Matrix<float, 2, 3> dx_dtranslation{
        dx_dX * body2camera_rotation_};
    Eigen::Matrix<float, 2, 6> dx_dtheta;
    dx_dtheta <<
        -dx_dtranslation * Vector2Skewsymmetric(data_point.center_f_body),
        dx_dtranslation;
    model_ecc_gradient_ -=
        (weight * residual_vector.transpose() * dx_dtheta).transpose();
    model_ecc_hessian_ -= weight * dx_dtheta.transpose() * dx_dtheta;
  }
  if (n_valid_model_ecc_points_ > 0)
    mean_model_ecc_residual_ /= float(n_valid_model_ecc_points_);

  for (const auto &data_point : video_ecc_data_points_) {
    const Eigen::Vector3f center_f_camera{
        body2camera_pose_ * data_point.center_f_body};
    const float x{center_f_camera(0)};
    const float y{center_f_camera(1)};
    const float z{center_f_camera(2)};
    if (z <= std::numeric_limits<float>::min()) continue;
    const Eigen::Vector2f center{x * fu_ / z + ppu_, y * fv_ / z + ppv_};
    const Eigen::Vector2f residual_vector{
        center - data_point.correspondence_center};
    const float residual{residual_vector.norm()};
    ++n_valid_video_ecc_points_;
    mean_video_ecc_residual_ += residual;
    if (residual >= video_ecc_tukey_constant_) continue;
    const float normalized_residual{residual / video_ecc_tukey_constant_};
    const float robust_factor{
        1.0f - normalized_residual * normalized_residual};
    const float weight{robust_factor * robust_factor / video_ecc_variance_};
    const float z2{z * z};
    Eigen::Matrix<float, 2, 3> dx_dX;
    dx_dX << fu_ / z, 0.0f, -x * fu_ / z2, 0.0f, fv_ / z,
        -y * fv_ / z2;
    const Eigen::Matrix<float, 2, 3> dx_dtranslation{
        dx_dX * body2camera_rotation_};
    Eigen::Matrix<float, 2, 6> dx_dtheta;
    dx_dtheta <<
        -dx_dtranslation * Vector2Skewsymmetric(data_point.center_f_body),
        dx_dtranslation;
    video_ecc_gradient_ -=
        (weight * residual_vector.transpose() * dx_dtheta).transpose();
    video_ecc_hessian_ -= weight * dx_dtheta.transpose() * dx_dtheta;
  }
  if (n_valid_video_ecc_points_ > 0)
    mean_video_ecc_residual_ /= float(n_valid_video_ecc_points_);
  if (collect_diagnostics_ && n_valid_gradient_points_ > 0) {
    mean_reprojection_error_ /= float(n_valid_gradient_points_);
    inlier_fraction_ /= float(n_valid_gradient_points_);
  }
  if (n_valid_model_texture_points_ > 0) {
    mean_model_texture_reprojection_error_ /=
        float(n_valid_model_texture_points_);
    model_texture_inlier_fraction_ /= float(n_valid_model_texture_points_);
  }
  if (n_valid_model_texture_disk_points_ > 0)
    mean_model_texture_disk_reprojection_error_ /=
        float(n_valid_model_texture_disk_points_);
  return true;
}

bool TextureModality::VisualizeOptimization(int save_idx) {
  if (!IsSetup()) return false;

  if (visualize_correspondences_optimization_)
    VisualizePointsColorImage("correspondences_optimization", save_idx);
  if (visualize_gradient_optimization_) VisualizeGradient();
  if (visualize_hessian_optimization_) VisualizeHessian();
  return true;
}

bool TextureModality::CalculateResults(int iteration) {
  if (!IsSetup()) return false;
  if (use_model_texture_matching_)
    model_texture_statistics_available_ = true;

  // Optical flow uses a strictly frame-to-frame reference. Reconstructing the
  // 3D anchors after every accepted tracking cycle prevents long-lived flow
  // drift from being interpreted as rigid object motion.
  if (use_optical_flow_tracking_) {
    PrecalculatePoseVariables();
    ComputeKeyframeData();
    return true;
  }

  // Calculate rotation difference
  Eigen::Vector3f orientation{
      body2camera_pose_.rotation().inverse() *
      body2camera_pose_.translation().matrix().normalized()};
  float rotation_difference =
      acos(orientation.transpose() * orientation_last_keyframe_);
  keyframe_age_++;

  // Compute new data if difference is above threshold
  if (rotation_difference > max_keyframe_rotation_difference_ ||
      keyframe_age_ > max_keyframe_age_)
    ComputeKeyframeData();
  return true;
}

bool TextureModality::VisualizeResults(int save_idx) {
  if (!IsSetup()) return false;

  if (keyframe_age_ == 0) {
    if (visualize_points_result_)
      VisualizePointsColorImage("color_image_result", save_idx);
    if (visualize_points_depth_image_result_ && measure_occlusions_)
      VisualizePointsDepthImage("depth_image_result", save_idx);
    if (visualize_points_depth_rendering_result_ && model_occlusions_)
      VisualizePointsDepthRendering("depth_rendering_result", save_idx);
    if (visualize_points_silhouette_rendering_result_)
      VisualizePointsSilhouetteRendering("silhouette_rendering_result",
                                         save_idx);
  }
  if (visualize_pose_result_) VisualizePose();
  return true;
}

const std::shared_ptr<ColorCamera> &TextureModality::color_camera_ptr() const {
  return color_camera_ptr_;
}

const std::shared_ptr<DepthCamera> &TextureModality::depth_camera_ptr() const {
  return depth_camera_ptr_;
}

const std::shared_ptr<FocusedSilhouetteRenderer>
    &TextureModality::silhouette_renderer_ptr() const {
  return silhouette_renderer_ptr_;
}

const std::shared_ptr<FocusedDepthRenderer>
    &TextureModality::depth_renderer_ptr() const {
  return depth_renderer_ptr_;
}

std::shared_ptr<Model> TextureModality::model_ptr() const { return {}; }

std::vector<std::shared_ptr<Camera>> TextureModality::camera_ptrs() const {
  return {color_camera_ptr_};
}

std::vector<std::shared_ptr<Renderer>>
TextureModality::start_modality_renderer_ptrs() const {
  return {silhouette_renderer_ptr_, depth_renderer_ptr_};
}

std::vector<std::shared_ptr<Renderer>> TextureModality::results_renderer_ptrs()
    const {
  return {silhouette_renderer_ptr_, depth_renderer_ptr_};
}

TextureModality::DescriptorType TextureModality::descriptor_type() const {
  return descriptor_type_;
}

int TextureModality::focused_image_size() const { return focused_image_size_; }

bool TextureModality::use_optical_flow_tracking() const {
  return use_optical_flow_tracking_;
}

bool TextureModality::use_geometry_guided_matching() const {
  return use_geometry_guided_matching_;
}

bool TextureModality::use_model_texture_matching() const {
  return use_model_texture_matching_;
}

bool TextureModality::use_model_photometric_tracking() const {
  return use_model_photometric_tracking_;
}

bool TextureModality::use_model_contour_tracking() const {
  return use_model_contour_tracking_;
}

bool TextureModality::use_model_ecc_tracking() const {
  return use_model_ecc_tracking_;
}

bool TextureModality::use_model_hypothesis_search() const {
  return use_model_hypothesis_search_;
}

bool TextureModality::use_video_ecc_tracking() const {
  return use_video_ecc_tracking_;
}

float TextureModality::descriptor_distance_threshold() const {
  return descriptor_distance_threshold_;
}

float TextureModality::tukey_norm_constant() const {
  return tukey_norm_constant_;
}

const std::vector<float> &TextureModality::standard_deviations() const {
  return standard_deviations_;
}

float TextureModality::max_keyframe_rotation_difference() const {
  return max_keyframe_rotation_difference_;
}

int TextureModality::max_keyframe_age() const { return max_keyframe_age_; }

int TextureModality::n_keyframes() const { return n_keyframes_; }

int TextureModality::orb_n_features() const { return orb_n_features_; }

float TextureModality::orb_scale_factor() const { return orb_scale_factor_; }

int TextureModality::orb_n_levels() const { return orb_n_levels_; }

int TextureModality::brisk_threshold() const { return brisk_threshold_; }

int TextureModality::brisk_octave() const { return brisk_octave_; }

float TextureModality::brisk_pattern_scale() const {
  return brisk_pattern_scale_;
}

float TextureModality::daisy_radius() const { return daisy_radius_; }

int TextureModality::daisy_q_radius() const { return daisy_q_radius_; }

int TextureModality::daisy_q_theta() const { return daisy_q_theta_; }

int TextureModality::daisy_q_hist() const { return daisy_q_hist_; }

bool TextureModality::freak_orientation_normalized() const {
  return freak_orientation_normalized_;
}

bool TextureModality::freak_scale_normalized() const {
  return freak_scale_normalized_;
}

float TextureModality::freak_pattern_scale() const {
  return freak_pattern_scale_;
}

int TextureModality::freak_n_octaves() const { return freak_n_octaves_; }

int TextureModality::sift_n_features() const { return sift_n_features_; }

int TextureModality::sift_n_octave_layers() const {
  return sift_n_octave_layers_;
}

double TextureModality::sift_contrast_threshold() const {
  return sift_contrast_threshold_;
}

double TextureModality::sift_edge_threshold() const {
  return sift_edge_threshold_;
}

double TextureModality::sift_sigma() const { return sift_sigma_; }

bool TextureModality::measure_occlusions() const { return measure_occlusions_; }

float TextureModality::measured_occlusion_radius() const {
  return measured_occlusion_radius_;
}

float TextureModality::measured_occlusion_threshold() const {
  return measured_occlusion_threshold_;
}

bool TextureModality::model_occlusions() const { return model_occlusions_; }

float TextureModality::modeled_occlusion_radius() const {
  return modeled_occlusion_radius_;
}

float TextureModality::modeled_occlusion_threshold() const {
  return modeled_occlusion_threshold_;
}

bool TextureModality::visualize_correspondences_correspondence() const {
  return visualize_correspondences_correspondence_;
}

bool TextureModality::visualize_points_depth_image_result() const {
  return visualize_points_depth_image_result_;
}

bool TextureModality::visualize_points_silhouette_rendering_result() const {
  return visualize_points_silhouette_rendering_result_;
}

bool TextureModality::visualize_points_depth_rendering_result() const {
  return visualize_points_depth_rendering_result_;
}

bool TextureModality::visualize_correspondences_optimization() const {
  return visualize_correspondences_optimization_;
}

bool TextureModality::visualize_points_result() const {
  return visualize_points_result_;
}

float TextureModality::visualization_min_depth() const {
  return visualization_min_depth_;
}

float TextureModality::visualization_max_depth() const {
  return visualization_max_depth_;
}

size_t TextureModality::n_correspondence_points() const {
  return data_points_.size();
}

int TextureModality::n_valid_gradient_points() const {
  return n_valid_gradient_points_;
}

float TextureModality::mean_reprojection_error() const {
  return mean_reprojection_error_;
}

float TextureModality::inlier_fraction() const { return inlier_fraction_; }

size_t TextureModality::n_model_texture_correspondence_points() const {
  return n_model_texture_correspondence_points_;
}

int TextureModality::n_valid_model_texture_points() const {
  return n_valid_model_texture_points_;
}

float TextureModality::mean_model_texture_reprojection_error() const {
  return mean_model_texture_reprojection_error_;
}

float TextureModality::model_texture_inlier_fraction() const {
  return model_texture_inlier_fraction_;
}

const Eigen::Matrix<float, 6, 1> &TextureModality::model_texture_gradient()
    const {
  return model_texture_gradient_;
}

const Eigen::Matrix<float, 6, 6> &TextureModality::model_texture_hessian()
    const {
  return model_texture_hessian_;
}

int TextureModality::n_valid_model_texture_orb_points() const {
  return n_valid_model_texture_points_ - n_valid_model_texture_disk_points_;
}

float TextureModality::mean_model_texture_orb_reprojection_error() const {
  const int n_orb_points{n_valid_model_texture_orb_points()};
  if (n_orb_points <= 0) return 0.0f;
  return (mean_model_texture_reprojection_error_ *
              float(n_valid_model_texture_points_) -
          mean_model_texture_disk_reprojection_error_ *
              float(n_valid_model_texture_disk_points_)) /
         float(n_orb_points);
}

Eigen::Matrix<float, 6, 1> TextureModality::model_texture_orb_gradient()
    const {
  return model_texture_gradient_ - model_texture_disk_gradient_;
}

Eigen::Matrix<float, 6, 6> TextureModality::model_texture_orb_hessian()
    const {
  return model_texture_hessian_ - model_texture_disk_hessian_;
}

int TextureModality::n_valid_model_texture_disk_points() const {
  return n_valid_model_texture_disk_points_;
}

float TextureModality::mean_model_texture_disk_reprojection_error() const {
  return mean_model_texture_disk_reprojection_error_;
}

const Eigen::Matrix<float, 6, 1> &
TextureModality::model_texture_disk_gradient() const {
  return model_texture_disk_gradient_;
}

const Eigen::Matrix<float, 6, 6> &
TextureModality::model_texture_disk_hessian() const {
  return model_texture_disk_hessian_;
}

int TextureModality::n_valid_model_photometric_points() const {
  return n_valid_model_photometric_points_;
}

float TextureModality::mean_model_photometric_residual() const {
  return mean_model_photometric_residual_;
}

const Eigen::Matrix<float, 6, 1> &
TextureModality::model_photometric_gradient() const {
  return model_photometric_gradient_;
}

const Eigen::Matrix<float, 6, 6> &
TextureModality::model_photometric_hessian() const {
  return model_photometric_hessian_;
}

int TextureModality::n_valid_model_contour_points() const {
  return n_valid_model_contour_points_;
}

float TextureModality::mean_model_contour_residual() const {
  return mean_model_contour_residual_;
}

const Eigen::Matrix<float, 6, 1> &TextureModality::model_contour_gradient()
    const {
  return model_contour_gradient_;
}

const Eigen::Matrix<float, 6, 6> &TextureModality::model_contour_hessian()
    const {
  return model_contour_hessian_;
}

int TextureModality::n_valid_model_ecc_points() const {
  return n_valid_model_ecc_points_;
}

float TextureModality::mean_model_ecc_residual() const {
  return mean_model_ecc_residual_;
}

float TextureModality::model_ecc_correlation() const {
  return model_ecc_correlation_;
}

const Eigen::Matrix<float, 6, 1> &TextureModality::model_ecc_gradient() const {
  return model_ecc_gradient_;
}

const Eigen::Matrix<float, 6, 6> &TextureModality::model_ecc_hessian() const {
  return model_ecc_hessian_;
}

float TextureModality::model_hypothesis_confidence() const {
  return model_hypothesis_confidence_;
}

const Eigen::Matrix<float, 6, 1> &
TextureModality::model_hypothesis_gradient() const {
  return model_hypothesis_gradient_;
}

const Eigen::Matrix<float, 6, 6> &
TextureModality::model_hypothesis_hessian() const {
  return model_hypothesis_hessian_;
}

int TextureModality::n_valid_video_ecc_points() const {
  return n_valid_video_ecc_points_;
}

float TextureModality::mean_video_ecc_residual() const {
  return mean_video_ecc_residual_;
}

float TextureModality::video_ecc_correlation() const {
  return video_ecc_correlation_;
}

const Eigen::Matrix<float, 6, 1> &TextureModality::video_ecc_gradient() const {
  return video_ecc_gradient_;
}

const Eigen::Matrix<float, 6, 6> &TextureModality::video_ecc_hessian() const {
  return video_ecc_hessian_;
}

bool TextureModality::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml file for general settings
  ReadOptionalValueFromYaml(fs, "descriptor_type", &descriptor_type_);
  ReadOptionalValueFromYaml(fs, "focused_image_size", &focused_image_size_);
  ReadOptionalValueFromYaml(fs, "descriptor_distance_threshold",
                            &descriptor_distance_threshold_);
  ReadOptionalValueFromYaml(fs, "tukey_norm_constant", &tukey_norm_constant_);
  ReadOptionalValueFromYaml(fs, "standard_deviations", &standard_deviations_);
  ReadOptionalValueFromYaml(fs, "max_keyframe_rotation_difference",
                            &max_keyframe_rotation_difference_);
  ReadOptionalValueFromYaml(fs, "max_keyframe_age", &max_keyframe_age_);
  ReadOptionalValueFromYaml(fs, "n_keyframes", &n_keyframes_);

  // Read parameters from yaml file for feature detection
  ReadOptionalValueFromYaml(fs, "orb_n_features", &orb_n_features_);
  ReadOptionalValueFromYaml(fs, "orb_scale_factor", &orb_scale_factor_);
  ReadOptionalValueFromYaml(fs, "orb_n_levels", &orb_n_levels_);
  ReadOptionalValueFromYaml(fs, "brisk_threshold", &brisk_threshold_);
  ReadOptionalValueFromYaml(fs, "brisk_octave", &brisk_octave_);
  ReadOptionalValueFromYaml(fs, "brisk_pattern_scale", &brisk_pattern_scale_);
  ReadOptionalValueFromYaml(fs, "daisy_radius", &daisy_radius_);
  ReadOptionalValueFromYaml(fs, "daisy_q_radius", &daisy_q_radius_);
  ReadOptionalValueFromYaml(fs, "daisy_q_theta", &daisy_q_theta_);
  ReadOptionalValueFromYaml(fs, "daisy_q_hist", &daisy_q_hist_);
  ReadOptionalValueFromYaml(fs, "freak_orientation_normalized",
                            &freak_orientation_normalized_);
  ReadOptionalValueFromYaml(fs, "freak_n_octaves", &freak_n_octaves_);
  ReadOptionalValueFromYaml(fs, "sift_n_features", &sift_n_features_);
  ReadOptionalValueFromYaml(fs, "sift_n_octave_layers", &sift_n_octave_layers_);
  ReadOptionalValueFromYaml(fs, "sift_contrast_threshold",
                            &sift_contrast_threshold_);
  ReadOptionalValueFromYaml(fs, "sift_edge_threshold", &sift_edge_threshold_);
  ReadOptionalValueFromYaml(fs, "sift_sigma", &sift_sigma_);

  // Read parameters from yaml file for occlusion handling
  ReadOptionalValueFromYaml(fs, "measured_occlusion_radius",
                            &measured_occlusion_radius_);
  ReadOptionalValueFromYaml(fs, "measured_occlusion_threshold",
                            &measured_occlusion_threshold_);
  ReadOptionalValueFromYaml(fs, "modeled_occlusion_radius",
                            &modeled_occlusion_radius_);
  ReadOptionalValueFromYaml(fs, "modeled_occlusion_threshold",
                            &modeled_occlusion_threshold_);

  // Read parameters from yaml for visualization
  ReadOptionalValueFromYaml(fs, "visualize_pose_result",
                            &visualize_pose_result_);
  ReadOptionalValueFromYaml(fs, "visualize_gradient_optimization",
                            &visualize_gradient_optimization_);
  ReadOptionalValueFromYaml(fs, "visualize_hessian_optimization",
                            &visualize_hessian_optimization_);
  ReadOptionalValueFromYaml(fs, "visualize_correspondences_correspondence",
                            &visualize_correspondences_correspondence_);
  ReadOptionalValueFromYaml(fs, "visualize_correspondences_optimization",
                            &visualize_correspondences_optimization_);
  ReadOptionalValueFromYaml(fs, "visualize_points_result",
                            &visualize_points_result_);
  ReadOptionalValueFromYaml(fs, "visualize_points_depth_image_result",
                            &visualize_points_depth_image_result_);
  ReadOptionalValueFromYaml(fs, "visualize_points_silhouette_rendering_result",
                            &visualize_points_depth_rendering_result_);
  ReadOptionalValueFromYaml(fs, "visualize_points_silhouette_rendering_result",
                            &visualize_points_depth_rendering_result_);
  ReadOptionalValueFromYaml(fs, "visualization_min_depth",
                            &visualization_min_depth_);
  ReadOptionalValueFromYaml(fs, "visualization_max_depth",
                            &visualization_max_depth_);
  ReadOptionalValueFromYaml(fs, "display_visualization",
                            &display_visualization_);
  ReadOptionalValueFromYaml(fs, "save_visualizations", &save_visualizations_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);

  // Process parameters
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  return true;
}

void TextureModality::SetUpFeatureDetectorAndMatcher() {
  // Set up feature detector
  switch (descriptor_type_) {
    case DescriptorType::BRISK:
      feature_detector_ = cv::BRISK::create(brisk_threshold_, brisk_octave_,
                                            brisk_pattern_scale_);
      feature_descriptor_ = cv::BRISK::create(brisk_threshold_, brisk_octave_,
                                              brisk_pattern_scale_);
      break;
    case DescriptorType::DAISY:
      feature_detector_ =
          cv::ORB::create(orb_n_features_, orb_scale_factor_, orb_n_levels_);
      feature_descriptor_ = cv::xfeatures2d::DAISY::create(
          daisy_radius_, daisy_q_radius_, daisy_q_theta_, daisy_q_hist_);
      break;
    case DescriptorType::FREAK:
      feature_detector_ =
          cv::ORB::create(orb_n_features_, orb_scale_factor_, orb_n_levels_);
      feature_descriptor_ = cv::xfeatures2d::FREAK::create(
          freak_orientation_normalized_, freak_scale_normalized_,
          freak_pattern_scale_, freak_n_octaves_);
      break;
    case DescriptorType::SIFT:
      // Look in both namespaces for SIFT
      using namespace cv;
      using namespace cv::xfeatures2d;
      feature_detector_ = SIFT::create(sift_n_features_, sift_n_octave_layers_,
                                       sift_contrast_threshold_,
                                       sift_edge_threshold_, sift_sigma_);
      feature_descriptor_ = SIFT::create(
          sift_n_features_, sift_n_octave_layers_, sift_contrast_threshold_,
          sift_edge_threshold_, sift_sigma_);
      break;
#ifdef USE_CUDA
    case DescriptorType::ORB_CUDA:
      feature_detector_orb_cuda_ = cv::cuda::ORB::create(
          orb_n_features_, orb_scale_factor_, orb_n_levels_);
#endif
    case DescriptorType::ORB:
    default:
      feature_detector_ =
          cv::ORB::create(orb_n_features_, orb_scale_factor_, orb_n_levels_);
      feature_descriptor_ =
          cv::ORB::create(orb_n_features_, orb_scale_factor_, orb_n_levels_);
      break;
  }

  // A dedicated, fixed-budget detector is used for synthetic-to-real CAD
  // correspondences. Extra pyramid levels and a lower FAST threshold improve
  // coverage without changing the tuned video-keyframe detector above.
  model_texture_feature_detector_ = cv::ORB::create(
      use_model_texture_disk_fusion_ ? 200 : 500, 1.2f, 4, 15, 0, 2,
      cv::ORB::HARRIS_SCORE, 31, 10);
  model_texture_sift_detector_ = cv::SIFT::create(300, 3, 0.02, 10.0, 1.2);
  model_texture_akaze_detector_ = cv::AKAZE::create(
      cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f, 4, 4,
      cv::KAZE::DIFF_PM_G2);

    // Set up feature matcher
#ifdef USE_CUDA
  if (descriptor_type_ == DescriptorType::ORB_CUDA)
    descriptor_matcher_cuda_ =
        cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  else
#endif
  {
    if (descriptor_type_ == DescriptorType::DAISY ||
        descriptor_type_ == DescriptorType::SIFT)
      descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_L2);
    else
      descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
  }
}

void TextureModality::SetImshowVariables() {
  imshow_correspondence_ =
      display_visualization_ && visualize_correspondences_correspondence_;
  imshow_optimization_ =
      display_visualization_ && visualize_correspondences_optimization_;
  imshow_result_ =
      display_visualization_ &&
      (visualize_points_result_ ||
       (visualize_points_depth_image_result_ && measure_occlusions_) ||
       (visualize_points_silhouette_rendering_result_) ||
       (visualize_points_depth_rendering_result_ && model_occlusions_));
}

void TextureModality::PrecalculateCameraVariables() {
  fu_ = color_camera_ptr_->intrinsics().fu;
  fv_ = color_camera_ptr_->intrinsics().fv;
  ppu_ = color_camera_ptr_->intrinsics().ppu;
  ppv_ = color_camera_ptr_->intrinsics().ppv;
  image_width_minus_1_ = color_camera_ptr_->intrinsics().width - 1;
  image_height_minus_1_ = color_camera_ptr_->intrinsics().height - 1;
  if (measure_occlusions_) {
    depth_fu_ = depth_camera_ptr_->intrinsics().fu;
    depth_fv_ = depth_camera_ptr_->intrinsics().fv;
    depth_ppu_ = depth_camera_ptr_->intrinsics().ppu;
    depth_ppv_ = depth_camera_ptr_->intrinsics().ppv;
    depth_scale_ = depth_camera_ptr_->depth_scale();
    depth_image_width_minus_1_ = depth_camera_ptr_->intrinsics().width - 1;
    depth_image_height_minus_1_ = depth_camera_ptr_->intrinsics().height - 1;
  }
}

void TextureModality::PrecalculateRendererVariables() {
  silhouette_image_size_minus_1_ = silhouette_renderer_ptr_->image_size() - 1;
  if (model_occlusions_) {
    depth_image_size_minus_1_ = depth_renderer_ptr_->image_size() - 1;
  }
}

void TextureModality::PrecalculatePoseVariables() {
  body2camera_pose_ =
      color_camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
  camera2body_pose_ = body2camera_pose_.inverse();
  body2camera_rotation_ = body2camera_pose_.rotation().matrix();
  if (measure_occlusions_) {
    body2depth_camera_pose_ =
        depth_camera_ptr_->world2camera_pose() * body_ptr_->body2world_pose();
  }
}

void TextureModality::PrecalculateIterationDependentVariables(
    int corr_iteration) {
  float standard_deviation =
      LastValidValue(standard_deviations_, corr_iteration);
  variance_ = powf(standard_deviation, 2.0f);
}

void TextureModality::DetectAndComputeCorrKeypoints() {
  keypoints_.clear();

  // Compute focused image
  cv::Mat focused_image;
  cv::Rect region_of_interest;
  float scale;
  if (!CalculateScaleAndRegionOfInterest(&region_of_interest, &scale)) return;
  cv::cvtColor(ObservedImage(), focused_image, cv::COLOR_BGR2GRAY);
  cv::resize(focused_image(region_of_interest), focused_image, cv::Size(),
             scale, scale);

  // Detect features and compute descriptors
#ifdef USE_CUDA
  if (descriptor_type_ == DescriptorType::ORB_CUDA) {
    cv::cuda::GpuMat focused_image_cuda{focused_image};
    feature_detector_orb_cuda_->detectAndCompute(
        focused_image_cuda, cv::noArray(), keypoints_, descriptors_cuda_);
  } else
#endif
  {
    feature_detector_->detect(focused_image, keypoints_);
    feature_descriptor_->compute(focused_image, keypoints_, descriptors_);
  }

  // Add focus offset to keypoints
  for (auto &corr_keypoint : keypoints_) {
    corr_keypoint.pt.x = region_of_interest.x + corr_keypoint.pt.x / scale;
    corr_keypoint.pt.y = region_of_interest.y + corr_keypoint.pt.y / scale;
  }
}

const cv::Mat &TextureModality::ObservedImage() const {
  if (use_unfiltered_observations_) {
    const auto loader =
        std::dynamic_pointer_cast<LoaderColorCamera>(color_camera_ptr_);
    if (loader && !loader->unfiltered_image().empty())
      return loader->unfiltered_image();
  }
  return color_camera_ptr_->image();
}

void TextureModality::ComputeOpticalFlowCorrespondences() {
  data_points_.clear();
  cv::Mat current_gray;
  cv::cvtColor(ObservedImage(), current_gray, cv::COLOR_BGR2GRAY);
  for (size_t i = 0; i < gray_images_keyframes_.size(); ++i) {
    if (gray_images_keyframes_[i].empty() ||
        image_points_keyframes_[i].empty())
      continue;

    std::vector<cv::Point2f> current_points;
    std::vector<cv::Point2f> backward_points;
    std::vector<uchar> forward_status;
    std::vector<uchar> backward_status;
    std::vector<float> forward_error;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
        gray_images_keyframes_[i], current_gray, image_points_keyframes_[i],
        current_points, forward_status, forward_error, cv::Size{21, 21}, 3,
        cv::TermCriteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                         0.01});
    cv::calcOpticalFlowPyrLK(
        current_gray, gray_images_keyframes_[i], current_points,
        backward_points, backward_status, backward_error, cv::Size{21, 21}, 3,
        cv::TermCriteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                         0.01});

    const auto &body_points{points_keyframes_[i]};
    const auto &keyframe_points{image_points_keyframes_[i]};
    for (size_t j = 0; j < current_points.size(); ++j) {
      if (!forward_status[j] || !backward_status[j]) continue;
      if (cv::norm(backward_points[j] - keyframe_points[j]) > 1.0f) continue;
      if (current_points[j].x < 0.0f ||
          current_points[j].x > float(image_width_minus_1_) ||
          current_points[j].y < 0.0f ||
          current_points[j].y > float(image_height_minus_1_))
        continue;

      DataPoint data_point;
      data_point.correspondence_center =
          Eigen::Vector2f{current_points[j].x, current_points[j].y};
      data_point.center_f_body = body_points[j];
      data_points_.push_back(std::move(data_point));
    }
  }

  // Forward-backward consistency only verifies image motion. A second robust
  // geometric check is required to ensure that tracks agree with one rigid
  // object pose before they enter the joint optimization.
  if (data_points_.size() < 6) {
    data_points_.clear();
    return;
  }
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(data_points_.size());
  image_points.reserve(data_points_.size());
  for (const auto &data_point : data_points_) {
    object_points.emplace_back(data_point.center_f_body(0),
                               data_point.center_f_body(1),
                               data_point.center_f_body(2));
    image_points.emplace_back(data_point.correspondence_center(0),
                              data_point.correspondence_center(1));
  }

  cv::Mat camera_matrix{
      (cv::Mat_<double>(3, 3) << fu_, 0.0, ppu_, 0.0, fv_, ppv_, 0.0, 0.0,
       1.0)};
  cv::Mat rotation_matrix(3, 3, CV_64F);
  cv::Mat translation_vector(3, 1, CV_64F);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col)
      rotation_matrix.at<double>(row, col) = body2camera_pose_(row, col);
    translation_vector.at<double>(row) = body2camera_pose_(row, 3);
  }
  cv::Mat rotation_vector;
  cv::Rodrigues(rotation_matrix, rotation_vector);
  std::vector<int> inlier_indexes;
  const bool pose_valid{cv::solvePnPRansac(
      object_points, image_points, camera_matrix, cv::noArray(),
      rotation_vector, translation_vector, true, 50, 2.0f, 0.99,
      inlier_indexes, cv::SOLVEPNP_ITERATIVE)};
  if (!pose_valid || inlier_indexes.size() < 6) {
    data_points_.clear();
    return;
  }

  std::vector<DataPoint> inlier_data_points;
  inlier_data_points.reserve(inlier_indexes.size());
  for (int index : inlier_indexes)
    inlier_data_points.push_back(data_points_[size_t(index)]);
  data_points_ = std::move(inlier_data_points);
}

void TextureModality::ComputeGeometryGuidedCorrespondences() {
  data_points_.clear();
#ifdef USE_CUDA
  if (descriptor_type_ == DescriptorType::ORB_CUDA) return;
#endif
  if (descriptors_.empty() || keypoints_.empty()) return;

  const int norm_type{descriptor_type_ == DescriptorType::DAISY ||
                              descriptor_type_ == DescriptorType::SIFT
                          ? cv::NORM_L2
                          : cv::NORM_HAMMING};
  const float search_radius_squared{
      tukey_norm_constant_ * tukey_norm_constant_};
  for (size_t keyframe_index = 0;
       keyframe_index < points_keyframes_.size(); ++keyframe_index) {
    const auto &body_points{points_keyframes_[keyframe_index]};
    const auto &keyframe_descriptors{
        descriptors_keyframes_[keyframe_index]};
    if (body_points.empty() || keyframe_descriptors.empty()) continue;

    std::vector<cv::Point2f> projected_points(body_points.size());
    for (size_t query_index = 0; query_index < body_points.size();
         ++query_index) {
      const Eigen::Vector3f point_f_camera{
          body2camera_pose_ * body_points[query_index]};
      projected_points[query_index] = cv::Point2f{
          point_f_camera(0) * fu_ / point_f_camera(2) + ppu_,
          point_f_camera(1) * fv_ / point_f_camera(2) + ppv_};
    }

    std::vector<int> query_to_train(body_points.size(), -1);
    std::vector<float> query_distance(body_points.size(),
                                      std::numeric_limits<float>::max());
    std::vector<float> query_second_distance(
        body_points.size(), std::numeric_limits<float>::max());
    std::vector<int> train_to_query(keypoints_.size(), -1);
    std::vector<float> train_distance(keypoints_.size(),
                                      std::numeric_limits<float>::max());
    for (size_t query_index = 0; query_index < body_points.size();
         ++query_index) {
      const auto &projected_point{projected_points[query_index]};
      for (size_t train_index = 0; train_index < keypoints_.size();
           ++train_index) {
        const cv::Point2f image_difference{
            keypoints_[train_index].pt - projected_point};
        if (image_difference.dot(image_difference) > search_radius_squared)
          continue;
        const float descriptor_distance{float(cv::norm(
            keyframe_descriptors.row(int(query_index)),
            descriptors_.row(int(train_index)), norm_type))};
        if (descriptor_distance < query_distance[query_index]) {
          query_second_distance[query_index] = query_distance[query_index];
          query_distance[query_index] = descriptor_distance;
          query_to_train[query_index] = int(train_index);
        } else if (descriptor_distance <
                   query_second_distance[query_index]) {
          query_second_distance[query_index] = descriptor_distance;
        }
        if (descriptor_distance < train_distance[train_index]) {
          train_distance[train_index] = descriptor_distance;
          train_to_query[train_index] = int(query_index);
        }
      }
    }

    for (size_t query_index = 0; query_index < body_points.size();
         ++query_index) {
      const int train_index{query_to_train[query_index]};
      if (train_index < 0 ||
          train_to_query[size_t(train_index)] != int(query_index))
        continue;
      if (!std::isfinite(query_second_distance[query_index]) ||
          query_distance[query_index] /
                  query_second_distance[query_index] >=
              descriptor_distance_threshold_)
        continue;
      DataPoint data_point;
      data_point.correspondence_center =
          Eigen::Vector2f{keypoints_[size_t(train_index)].pt.x,
                          keypoints_[size_t(train_index)].pt.y};
      data_point.center_f_body = body_points[query_index];
      data_points_.push_back(std::move(data_point));
    }
  }
}

void TextureModality::ComputeModelTextureCorrespondences() {
  n_model_texture_correspondence_points_ = 0;
#ifdef USE_ONNXRUNTIME
  if (use_adaptive_model_texture_disk_ &&
      model_texture_statistics_available_) {
    if (adaptive_model_texture_disk_selected_) {
      if (n_valid_model_texture_points_ > 0 &&
          mean_model_texture_reprojection_error_ > 8.0f) {
        adaptive_model_texture_disk_selected_ = false;
        adaptive_model_texture_disk_evidence_ = 0;
      }
    } else {
      const bool disk_candidate{
          n_valid_model_texture_points_ < 30 &&
          (n_valid_model_texture_points_ == 0 ||
           mean_model_texture_reprojection_error_ < 5.0f)};
      adaptive_model_texture_disk_evidence_ =
          disk_candidate
              ? std::min(3, adaptive_model_texture_disk_evidence_ + 1)
              : 0;
      if (adaptive_model_texture_disk_evidence_ >= 3)
        adaptive_model_texture_disk_selected_ = true;
    }
  }
  if (use_model_texture_disk_ || use_model_texture_disk_fusion_ ||
      adaptive_model_texture_disk_selected_) {
    ComputeDiskModelTextureCorrespondences();
    if (!use_model_texture_disk_fusion_) return;
  }
#endif
#ifdef USE_CUDA
  if (descriptor_type_ == DescriptorType::ORB_CUDA) return;
#endif
  if (!body_ptr_->has_texture()) return;

  cv::Mat rendered_gray;
  cv::cvtColor(textured_renderer_ptr_->focused_color_image(), rendered_gray,
               cv::COLOR_BGRA2GRAY);
  cv::equalizeHist(rendered_gray, rendered_gray);
  std::vector<cv::KeyPoint> rendered_keypoints;
  cv::Mat rendered_descriptors;
  const cv::Ptr<cv::Feature2D> &model_feature_detector{
      use_model_texture_sift_
          ? model_texture_sift_detector_
          : (use_model_texture_akaze_ ? model_texture_akaze_detector_
                                      : model_texture_feature_detector_)};
  model_feature_detector->detectAndCompute(
      rendered_gray, cv::noArray(), rendered_keypoints, rendered_descriptors);

  cv::Rect current_roi;
  float current_scale;
  if (!CalculateScaleAndRegionOfInterest(&current_roi, &current_scale)) return;
  cv::Mat current_gray;
  cv::cvtColor(ObservedImage(), current_gray, cv::COLOR_BGR2GRAY);
  cv::resize(current_gray(current_roi), current_gray, cv::Size(), current_scale,
             current_scale);
  if (!model_texture_debug_directory_.empty() &&
      !model_texture_debug_images_saved_) {
    std::filesystem::create_directories(model_texture_debug_directory_);
    cv::imwrite((model_texture_debug_directory_ /
                 (body_ptr_->name() + "_cad_render.png"))
                    .string(),
                textured_renderer_ptr_->focused_color_image());
    cv::imwrite((model_texture_debug_directory_ /
                 (body_ptr_->name() + "_observed_crop.png"))
                    .string(),
                current_gray);
    model_texture_debug_images_saved_ = true;
  }
  cv::equalizeHist(current_gray, current_gray);
  std::vector<cv::KeyPoint> current_keypoints;
  cv::Mat current_descriptors;
  model_feature_detector->detectAndCompute(
      current_gray, cv::noArray(), current_keypoints, current_descriptors);
  for (auto &keypoint : current_keypoints) {
    keypoint.pt.x = current_roi.x + keypoint.pt.x / current_scale;
    keypoint.pt.y = current_roi.y + keypoint.pt.y / current_scale;
  }
  if (rendered_descriptors.empty() || rendered_keypoints.empty() ||
      current_descriptors.empty() || current_keypoints.empty())
    return;

  const int norm_type{use_model_texture_sift_ ? cv::NORM_L2
                                              : cv::NORM_HAMMING};
  const float radius_squared{model_texture_search_radius_ *
                             model_texture_search_radius_};
  std::vector<int> query_to_train(rendered_keypoints.size(), -1);
  std::vector<float> query_distance(rendered_keypoints.size(),
                                    std::numeric_limits<float>::max());
  std::vector<float> query_second_distance(rendered_keypoints.size(),
                                           std::numeric_limits<float>::max());
  std::vector<int> train_to_query(current_keypoints.size(), -1);
  std::vector<float> train_distance(current_keypoints.size(),
                                    std::numeric_limits<float>::max());

  const float corner_u{textured_renderer_ptr_->corner_u()};
  const float corner_v{textured_renderer_ptr_->corner_v()};
  const float inverse_scale{1.0f / textured_renderer_ptr_->scale()};
  for (size_t query_index = 0; query_index < rendered_keypoints.size();
       ++query_index) {
    const cv::Point2f projected_point{
        corner_u + rendered_keypoints[query_index].pt.x * inverse_scale,
        corner_v + rendered_keypoints[query_index].pt.y * inverse_scale};
    for (size_t train_index = 0; train_index < current_keypoints.size();
         ++train_index) {
      const cv::Point2f difference{current_keypoints[train_index].pt -
                                   projected_point};
      if (difference.dot(difference) > radius_squared) continue;
      const float descriptor_distance{float(cv::norm(
          rendered_descriptors.row(int(query_index)),
          current_descriptors.row(int(train_index)), norm_type))};
      if (descriptor_distance < query_distance[query_index]) {
        query_second_distance[query_index] = query_distance[query_index];
        query_distance[query_index] = descriptor_distance;
        query_to_train[query_index] = int(train_index);
      } else if (descriptor_distance < query_second_distance[query_index]) {
        query_second_distance[query_index] = descriptor_distance;
      }
      if (descriptor_distance < train_distance[train_index]) {
        train_distance[train_index] = descriptor_distance;
        train_to_query[train_index] = int(query_index);
      }
    }
  }

  const cv::Mat &color_image{textured_renderer_ptr_->focused_color_image()};
  const cv::Mat &depth_image{textured_renderer_ptr_->focused_depth_image()};
  for (size_t query_index = 0; query_index < rendered_keypoints.size();
       ++query_index) {
    const int train_index{query_to_train[query_index]};
    if (train_index < 0 ||
        train_to_query[size_t(train_index)] != int(query_index) ||
        (!use_model_texture_sift_ &&
         query_distance[query_index] >
             model_texture_max_descriptor_distance_) ||
        !std::isfinite(query_second_distance[query_index]) ||
        query_distance[query_index] / query_second_distance[query_index] >=
            model_texture_ratio_threshold_)
      continue;

    const int u{int(rendered_keypoints[query_index].pt.x + 0.5f)};
    const int v{int(rendered_keypoints[query_index].pt.y + 0.5f)};
    if (u < 0 || u >= color_image.cols || v < 0 || v >= color_image.rows ||
        color_image.at<cv::Vec4b>(v, u)[3] == 0)
      continue;
    const ushort depth_value{depth_image.at<ushort>(v, u)};
    if (depth_value == std::numeric_limits<ushort>::max()) continue;
    const float center_u{corner_u + float(u) * inverse_scale};
    const float center_v{corner_v + float(v) * inverse_scale};
    const float depth{textured_renderer_ptr_->Depth(depth_value)};
    const Eigen::Vector3f center_f_camera{
        depth * (center_u - ppu_) / fu_, depth * (center_v - ppv_) / fv_,
        depth};

    DataPoint data_point;
    data_point.from_model_texture = true;
    data_point.center_f_body = camera2body_pose_ * center_f_camera;
    data_point.correspondence_center =
        Eigen::Vector2f{current_keypoints[size_t(train_index)].pt.x,
                        current_keypoints[size_t(train_index)].pt.y};
    data_points_.push_back(std::move(data_point));
    ++n_model_texture_correspondence_points_;
  }
}

void TextureModality::ComputeDiskModelTextureCorrespondences() {
#ifdef USE_ONNXRUNTIME
  if (!disk_feature_extractor_ || !disk_feature_extractor_->set_up() ||
      !body_ptr_->has_texture())
    return;

  const cv::Mat &rendered_color{
      textured_renderer_ptr_->focused_color_image()};
  const cv::Mat &rendered_depth{
      textured_renderer_ptr_->focused_depth_image()};
  std::vector<cv::KeyPoint> rendered_keypoints;
  cv::Mat rendered_descriptors;
  if (!disk_feature_extractor_->DetectAndCompute(
          rendered_color, &rendered_keypoints, &rendered_descriptors))
    return;

  cv::Rect current_roi;
  float unused_scale;
  if (!CalculateScaleAndRegionOfInterest(&current_roi, &unused_scale)) return;
  const cv::Mat current_crop{ObservedImage()(current_roi)};
  std::vector<cv::KeyPoint> current_keypoints;
  cv::Mat current_descriptors;
  if (!disk_feature_extractor_->DetectAndCompute(
          current_crop, &current_keypoints, &current_descriptors))
    return;
  if (rendered_descriptors.empty() || current_descriptors.empty()) return;

  const float input_size{float(disk_feature_extractor_->input_size())};
  const float rendered_scale_u{float(rendered_color.cols) / input_size};
  const float rendered_scale_v{float(rendered_color.rows) / input_size};
  const float current_scale_u{float(current_roi.width) / input_size};
  const float current_scale_v{float(current_roi.height) / input_size};
  for (auto &keypoint : rendered_keypoints) {
    keypoint.pt.x *= rendered_scale_u;
    keypoint.pt.y *= rendered_scale_v;
  }
  for (auto &keypoint : current_keypoints) {
    keypoint.pt.x = current_roi.x + keypoint.pt.x * current_scale_u;
    keypoint.pt.y = current_roi.y + keypoint.pt.y * current_scale_v;
  }

  const float corner_u{textured_renderer_ptr_->corner_u()};
  const float corner_v{textured_renderer_ptr_->corner_v()};
  const float inverse_scale{1.0f / textured_renderer_ptr_->scale()};
  const float radius{1.5f * model_texture_search_radius_};
  const float radius_squared{radius * radius};
  std::vector<int> query_to_train(rendered_keypoints.size(), -1);
  std::vector<float> query_similarity(rendered_keypoints.size(), -1.0f);
  std::vector<int> train_to_query(current_keypoints.size(), -1);
  std::vector<float> train_similarity(current_keypoints.size(), -1.0f);
  for (size_t query_index = 0; query_index < rendered_keypoints.size();
       ++query_index) {
    const cv::Point2f projected_point{
        corner_u + rendered_keypoints[query_index].pt.x * inverse_scale,
        corner_v + rendered_keypoints[query_index].pt.y * inverse_scale};
    for (size_t train_index = 0; train_index < current_keypoints.size();
         ++train_index) {
      const cv::Point2f difference{current_keypoints[train_index].pt -
                                   projected_point};
      if (difference.dot(difference) > radius_squared) continue;
      const float similarity{float(
          rendered_descriptors.row(int(query_index))
              .dot(current_descriptors.row(int(train_index))))};
      if (similarity > query_similarity[query_index]) {
        query_similarity[query_index] = similarity;
        query_to_train[query_index] = int(train_index);
      }
      if (similarity > train_similarity[train_index]) {
        train_similarity[train_index] = similarity;
        train_to_query[train_index] = int(query_index);
      }
    }
  }

  const auto is_descriptor_match = [&](size_t query_index) {
    const int train_index{query_to_train[query_index]};
    return train_index >= 0 &&
           train_to_query[size_t(train_index)] == int(query_index) &&
           query_similarity[query_index] >=
               model_texture_disk_similarity_threshold_;
  };
  std::vector<uchar> geometric_inliers(rendered_keypoints.size(), uchar{1});
  std::vector<size_t> candidate_queries;
  std::vector<cv::Point2f> projected_points;
  std::vector<cv::Point2f> observed_points;
  for (size_t query_index = 0; query_index < rendered_keypoints.size();
       ++query_index) {
    if (!is_descriptor_match(query_index)) continue;
    candidate_queries.push_back(query_index);
    projected_points.emplace_back(
        corner_u + rendered_keypoints[query_index].pt.x * inverse_scale,
        corner_v + rendered_keypoints[query_index].pt.y * inverse_scale);
    observed_points.push_back(
        current_keypoints[size_t(query_to_train[query_index])].pt);
  }
  if (candidate_queries.size() >= 4) {
    cv::Mat inlier_mask;
    cv::Mat affine;
    if (use_model_texture_pnp_geometry_) {
      std::vector<cv::Point3f> body_points;
      std::vector<cv::Point2f> valid_observed_points;
      std::vector<size_t> valid_candidate_indexes;
      body_points.reserve(candidate_queries.size());
      valid_observed_points.reserve(candidate_queries.size());
      valid_candidate_indexes.reserve(candidate_queries.size());
      for (size_t candidate_index = 0;
           candidate_index < candidate_queries.size(); ++candidate_index) {
        const size_t query_index{candidate_queries[candidate_index]};
        const int u{int(rendered_keypoints[query_index].pt.x + 0.5f)};
        const int v{int(rendered_keypoints[query_index].pt.y + 0.5f)};
        if (u < 0 || u >= rendered_depth.cols || v < 0 ||
            v >= rendered_depth.rows)
          continue;
        const ushort depth_value{rendered_depth.at<ushort>(v, u)};
        if (depth_value == std::numeric_limits<ushort>::max()) continue;
        const float center_u{corner_u + float(u) * inverse_scale};
        const float center_v{corner_v + float(v) * inverse_scale};
        const float depth{textured_renderer_ptr_->Depth(depth_value)};
        const Eigen::Vector3f center_f_camera{
            depth * (center_u - ppu_) / fu_,
            depth * (center_v - ppv_) / fv_, depth};
        const Eigen::Vector3f center_f_body{
            camera2body_pose_ * center_f_camera};
        body_points.emplace_back(center_f_body.x(), center_f_body.y(),
                                 center_f_body.z());
        valid_observed_points.push_back(observed_points[candidate_index]);
        valid_candidate_indexes.push_back(candidate_index);
      }
      if (body_points.size() >= 4) {
        cv::Mat camera_matrix{
            (cv::Mat_<double>(3, 3) << fu_, 0.0, ppu_, 0.0, fv_, ppv_, 0.0,
             0.0, 1.0)};
        cv::Mat rotation_matrix(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row)
          for (int column = 0; column < 3; ++column)
            rotation_matrix.at<double>(row, column) =
                body2camera_pose_.rotation()(row, column);
        cv::Mat rotation_vector;
        cv::Rodrigues(rotation_matrix, rotation_vector);
        cv::Mat translation_vector{
            (cv::Mat_<double>(3, 1)
             << body2camera_pose_.translation().x(),
                body2camera_pose_.translation().y(),
                body2camera_pose_.translation().z())};
        cv::Mat pnp_inliers;
        const bool pnp_success{cv::solvePnPRansac(
            body_points, valid_observed_points, camera_matrix, cv::noArray(),
            rotation_vector, translation_vector, true, 100,
            model_texture_disk_ransac_threshold_, 0.99, pnp_inliers,
            cv::SOLVEPNP_EPNP)};
        if (pnp_success && pnp_inliers.rows >= 3) {
          inlier_mask = cv::Mat::zeros(int(candidate_queries.size()), 1, CV_8U);
          for (int row = 0; row < pnp_inliers.rows; ++row) {
            const size_t valid_index{
                size_t(pnp_inliers.at<int>(row, 0))};
            inlier_mask.at<uchar>(int(valid_candidate_indexes[valid_index]), 0) =
                uchar{1};
          }
          affine = cv::Mat::eye(2, 3, CV_64F);
        }
      }
      if (affine.empty()) {
        affine = cv::estimateAffine2D(
            projected_points, observed_points, inlier_mask, cv::RANSAC,
            model_texture_disk_ransac_threshold_, 500, 0.99, 5);
      }
    } else if (use_adaptive_model_texture_geometry_) {
      cv::Mat full_mask;
      cv::Mat partial_mask;
      const cv::Mat full_affine{cv::estimateAffine2D(
          projected_points, observed_points, full_mask, cv::RANSAC,
          model_texture_disk_ransac_threshold_, 500, 0.99, 5)};
      const cv::Mat partial_affine{cv::estimateAffinePartial2D(
          projected_points, observed_points, partial_mask, cv::RANSAC,
          model_texture_disk_ransac_threshold_, 500, 0.99, 5)};
      const int n_full_inliers{
          full_affine.empty() ? 0 : cv::countNonZero(full_mask)};
      const int n_partial_inliers{
          partial_affine.empty() ? 0 : cv::countNonZero(partial_mask)};
      if (n_full_inliers >= 3 && n_partial_inliers >= 3) {
        cv::bitwise_and(full_mask, partial_mask, inlier_mask);
        if (cv::countNonZero(inlier_mask) >= 3) {
          affine = full_affine;
        } else if (n_full_inliers >= n_partial_inliers) {
          affine = full_affine;
          inlier_mask = full_mask;
        } else {
          affine = partial_affine;
          inlier_mask = partial_mask;
        }
      } else if (n_full_inliers >= n_partial_inliers) {
        affine = full_affine;
        inlier_mask = full_mask;
      } else {
        affine = partial_affine;
        inlier_mask = partial_mask;
      }
    } else {
      affine = cv::estimateAffine2D(
          projected_points, observed_points, inlier_mask, cv::RANSAC,
          model_texture_disk_ransac_threshold_, 500, 0.99, 5);
    }
    if (!affine.empty() && cv::countNonZero(inlier_mask) >= 3) {
      std::fill(geometric_inliers.begin(), geometric_inliers.end(), uchar{0});
      for (size_t match_index = 0; match_index < candidate_queries.size();
           ++match_index) {
        if (inlier_mask.at<uchar>(int(match_index), 0))
          geometric_inliers[candidate_queries[match_index]] = uchar{1};
      }
    }
  }

  for (size_t query_index = 0; query_index < rendered_keypoints.size();
       ++query_index) {
    const int train_index{query_to_train[query_index]};
    if (!is_descriptor_match(query_index) ||
        geometric_inliers[query_index] == 0)
      continue;

    const int u{int(rendered_keypoints[query_index].pt.x + 0.5f)};
    const int v{int(rendered_keypoints[query_index].pt.y + 0.5f)};
    if (u < 0 || u >= rendered_color.cols || v < 0 ||
        v >= rendered_color.rows || rendered_color.at<cv::Vec4b>(v, u)[3] == 0)
      continue;
    const ushort depth_value{rendered_depth.at<ushort>(v, u)};
    if (depth_value == std::numeric_limits<ushort>::max()) continue;
    const float center_u{corner_u + float(u) * inverse_scale};
    const float center_v{corner_v + float(v) * inverse_scale};
    const float depth{textured_renderer_ptr_->Depth(depth_value)};
    const Eigen::Vector3f center_f_camera{
        depth * (center_u - ppu_) / fu_, depth * (center_v - ppv_) / fv_,
        depth};

    DataPoint data_point;
    data_point.from_model_texture = true;
    data_point.from_model_texture_disk = true;
    data_point.center_f_body = camera2body_pose_ * center_f_camera;
    data_point.correspondence_center =
        Eigen::Vector2f{current_keypoints[size_t(train_index)].pt.x,
                        current_keypoints[size_t(train_index)].pt.y};
    data_points_.push_back(std::move(data_point));
    ++n_model_texture_correspondence_points_;
  }
#endif
}

void TextureModality::ComputeModelPhotometricData() {
  model_photometric_data_points_.clear();
  model_photometric_scale_ = 1.0f;
  model_photometric_offset_ = 0.0f;

  const cv::Mat &rendered_color{
      textured_renderer_ptr_->focused_color_image()};
  const cv::Mat &rendered_depth{
      textured_renderer_ptr_->focused_depth_image()};
  cv::Mat rendered_gray;
  cv::cvtColor(rendered_color, rendered_gray, cv::COLOR_BGRA2GRAY);
  cv::Mat rendered_gradient_x;
  cv::Mat rendered_gradient_y;
  cv::Sobel(rendered_gray, rendered_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(rendered_gray, rendered_gradient_y, CV_32F, 0, 1, 3, 0.125);

  // Suppress the silhouette boundary so the direct term contains texture
  // evidence rather than duplicating the region modality.
  cv::Mat alpha;
  cv::extractChannel(rendered_color, alpha, 3);
  cv::threshold(alpha, alpha, 0, 255, cv::THRESH_BINARY);
  cv::erode(alpha, alpha,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{7, 7}));

  struct Candidate {
    int u = 0;
    int v = 0;
    float score = -1.0f;
  };
  constexpr int cell_size{10};
  const int n_cells_u{(rendered_gray.cols + cell_size - 1) / cell_size};
  const int n_cells_v{(rendered_gray.rows + cell_size - 1) / cell_size};
  std::vector<Candidate> candidates(size_t(n_cells_u * n_cells_v));
  for (int v = 1; v < rendered_gray.rows - 1; ++v) {
    for (int u = 1; u < rendered_gray.cols - 1; ++u) {
      if (alpha.at<uchar>(v, u) == 0) continue;
      const float gx{rendered_gradient_x.at<float>(v, u)};
      const float gy{rendered_gradient_y.at<float>(v, u)};
      const float score{gx * gx + gy * gy};
      if (score < 64.0f) continue;
      Candidate &candidate{
          candidates[size_t((v / cell_size) * n_cells_u + u / cell_size)]};
      if (score > candidate.score) candidate = Candidate{u, v, score};
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.score > b.score;
            });

  cv::Mat measured_gray_8u;
  cv::cvtColor(ObservedImage(), measured_gray_8u,
               cv::COLOR_BGR2GRAY);
  const float corner_u{textured_renderer_ptr_->corner_u()};
  const float corner_v{textured_renderer_ptr_->corner_v()};
  const float inverse_scale{1.0f / textured_renderer_ptr_->scale()};
  std::vector<float> model_intensities;
  std::vector<float> measured_intensities;
  model_intensities.reserve(size_t(n_model_photometric_points_));
  measured_intensities.reserve(size_t(n_model_photometric_points_));
  for (const auto &candidate : candidates) {
    if (candidate.score < 0.0f ||
        int(model_photometric_data_points_.size()) >=
            n_model_photometric_points_)
      break;
    const ushort depth_value{
        rendered_depth.at<ushort>(candidate.v, candidate.u)};
    if (depth_value == std::numeric_limits<ushort>::max()) continue;
    const float center_u{corner_u + float(candidate.u) * inverse_scale};
    const float center_v{corner_v + float(candidate.v) * inverse_scale};
    const int measured_u{int(center_u + 0.5f)};
    const int measured_v{int(center_v + 0.5f)};
    if (measured_u < 0 || measured_u >= measured_gray_8u.cols ||
        measured_v < 0 || measured_v >= measured_gray_8u.rows)
      continue;
    const float depth{textured_renderer_ptr_->Depth(depth_value)};
    const Eigen::Vector3f center_f_camera{
        depth * (center_u - ppu_) / fu_, depth * (center_v - ppv_) / fv_,
        depth};
    const float model_intensity{
        float(rendered_gray.at<uchar>(candidate.v, candidate.u))};
    model_photometric_data_points_.push_back(
        PhotometricDataPoint{camera2body_pose_ * center_f_camera,
                             model_intensity});
    model_intensities.push_back(model_intensity);
    measured_intensities.push_back(
        float(measured_gray_8u.at<uchar>(measured_v, measured_u)));
  }

  if (model_intensities.size() >= 6) {
    float model_mean{0.0f};
    float measured_mean{0.0f};
    for (size_t i = 0; i < model_intensities.size(); ++i) {
      model_mean += model_intensities[i];
      measured_mean += measured_intensities[i];
    }
    model_mean /= float(model_intensities.size());
    measured_mean /= float(measured_intensities.size());
    float model_variance{0.0f};
    float measured_variance{0.0f};
    for (size_t i = 0; i < model_intensities.size(); ++i) {
      model_variance +=
          (model_intensities[i] - model_mean) *
          (model_intensities[i] - model_mean);
      measured_variance +=
          (measured_intensities[i] - measured_mean) *
          (measured_intensities[i] - measured_mean);
    }
    model_photometric_scale_ = std::clamp(
        std::sqrt(measured_variance / std::max(1.0f, model_variance)), 0.5f,
        2.0f);
    model_photometric_offset_ =
        measured_mean - model_photometric_scale_ * model_mean;
  }
  measured_gray_8u.convertTo(model_photometric_gray_image_, CV_32F);
  cv::Sobel(model_photometric_gray_image_, model_photometric_gradient_x_,
            CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(model_photometric_gray_image_, model_photometric_gradient_y_,
            CV_32F, 0, 1, 3, 0.125);
}

void TextureModality::ComputeModelContourData() {
  model_contour_data_points_.clear();
  const cv::Mat &rendered_color{
      textured_renderer_ptr_->focused_color_image()};
  const cv::Mat &rendered_depth{
      textured_renderer_ptr_->focused_depth_image()};
  cv::Mat rendered_gray;
  cv::cvtColor(rendered_color, rendered_gray, cv::COLOR_BGRA2GRAY);
  cv::Mat rendered_gradient_x;
  cv::Mat rendered_gradient_y;
  cv::Sobel(rendered_gray, rendered_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(rendered_gray, rendered_gradient_y, CV_32F, 0, 1, 3, 0.125);

  cv::Mat alpha;
  cv::extractChannel(rendered_color, alpha, 3);
  cv::threshold(alpha, alpha, 0, 255, cv::THRESH_BINARY);
  cv::erode(alpha, alpha,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{7, 7}));

  struct Candidate {
    int u = 0;
    int v = 0;
    float score = -1.0f;
    Eigen::Vector2f normal{};
  };
  constexpr int cell_size{8};
  const int n_cells_u{(rendered_gray.cols + cell_size - 1) / cell_size};
  const int n_cells_v{(rendered_gray.rows + cell_size - 1) / cell_size};
  std::vector<Candidate> candidates(size_t(n_cells_u * n_cells_v));
  for (int v = 1; v < rendered_gray.rows - 1; ++v) {
    for (int u = 1; u < rendered_gray.cols - 1; ++u) {
      if (alpha.at<uchar>(v, u) == 0) continue;
      const Eigen::Vector2f gradient{
          rendered_gradient_x.at<float>(v, u),
          rendered_gradient_y.at<float>(v, u)};
      const float magnitude{gradient.norm()};
      if (magnitude < 8.0f) continue;
      Candidate &candidate{
          candidates[size_t((v / cell_size) * n_cells_u + u / cell_size)]};
      if (magnitude > candidate.score)
        candidate = Candidate{u, v, magnitude, gradient / magnitude};
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.score > b.score;
            });

  cv::Mat measured_gray;
  cv::cvtColor(ObservedImage(), measured_gray, cv::COLOR_BGR2GRAY);
  cv::Mat measured_gradient_x;
  cv::Mat measured_gradient_y;
  cv::Sobel(measured_gray, measured_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(measured_gray, measured_gradient_y, CV_32F, 0, 1, 3, 0.125);
  const float corner_u{textured_renderer_ptr_->corner_u()};
  const float corner_v{textured_renderer_ptr_->corner_v()};
  const float inverse_scale{1.0f / textured_renderer_ptr_->scale()};
  const int search_radius{int(model_contour_search_radius_ + 0.5f)};
  for (const auto &candidate : candidates) {
    if (candidate.score < 0.0f ||
        int(model_contour_data_points_.size()) >= n_model_contour_points_)
      break;
    const float center_u{corner_u + float(candidate.u) * inverse_scale};
    const float center_v{corner_v + float(candidate.v) * inverse_scale};
    float best_score{0.0f};
    float best_displacement{0.0f};
    for (int displacement = -search_radius; displacement <= search_radius;
         ++displacement) {
      const float u{center_u + float(displacement) * candidate.normal(0)};
      const float v{center_v + float(displacement) * candidate.normal(1)};
      const int measured_u{int(u + 0.5f)};
      const int measured_v{int(v + 0.5f)};
      if (measured_u < 1 || measured_u >= measured_gray.cols - 1 ||
          measured_v < 1 || measured_v >= measured_gray.rows - 1)
        continue;
      const Eigen::Vector2f measured_gradient{
          measured_gradient_x.at<float>(measured_v, measured_u),
          measured_gradient_y.at<float>(measured_v, measured_u)};
      const float magnitude{measured_gradient.norm()};
      if (magnitude < 8.0f) continue;
      const float alignment{
          std::abs(candidate.normal.dot(measured_gradient / magnitude))};
      if (alignment < 0.65f) continue;
      const float score{magnitude * alignment * alignment /
                        (1.0f + 0.1f * std::abs(float(displacement)))};
      if (score > best_score) {
        best_score = score;
        best_displacement = float(displacement);
      }
    }
    if (best_score <= 0.0f) continue;

    const ushort depth_value{
        rendered_depth.at<ushort>(candidate.v, candidate.u)};
    if (depth_value == std::numeric_limits<ushort>::max()) continue;
    const float depth{textured_renderer_ptr_->Depth(depth_value)};
    const Eigen::Vector3f center_f_camera{
        depth * (center_u - ppu_) / fu_, depth * (center_v - ppv_) / fv_,
        depth};
    const Eigen::Vector2f projected_center{center_u, center_v};
    model_contour_data_points_.push_back(ModelContourDataPoint{
        camera2body_pose_ * center_f_camera, candidate.normal,
        projected_center + best_displacement * candidate.normal});
  }
}

void TextureModality::ComputeModelEccData() {
  model_ecc_data_points_.clear();
  model_ecc_correlation_ = 0.0f;
  const cv::Mat &rendered_color{
      textured_renderer_ptr_->focused_color_image()};
  const cv::Mat &rendered_depth{
      textured_renderer_ptr_->focused_depth_image()};
  cv::Mat rendered_gray;
  cv::cvtColor(rendered_color, rendered_gray, cv::COLOR_BGRA2GRAY);
  cv::Mat rendered_gradient_x;
  cv::Mat rendered_gradient_y;
  cv::Sobel(rendered_gray, rendered_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(rendered_gray, rendered_gradient_y, CV_32F, 0, 1, 3, 0.125);
  cv::Mat rendered_magnitude;
  cv::magnitude(rendered_gradient_x, rendered_gradient_y, rendered_magnitude);

  cv::Mat alpha;
  cv::extractChannel(rendered_color, alpha, 3);
  cv::threshold(alpha, alpha, 0, 255, cv::THRESH_BINARY);
  cv::erode(alpha, alpha,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{7, 7}));

  cv::Mat measured_gray;
  cv::cvtColor(ObservedImage(), measured_gray, cv::COLOR_BGR2GRAY);
  cv::Mat measured_focused;
  const float inverse_scale{1.0f / textured_renderer_ptr_->scale()};
  const cv::Mat focus_mapping{
      (cv::Mat_<float>(2, 3) << inverse_scale, 0.0f,
       textured_renderer_ptr_->corner_u(), 0.0f, inverse_scale,
       textured_renderer_ptr_->corner_v())};
  cv::warpAffine(measured_gray, measured_focused, focus_mapping,
                 rendered_gray.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT);
  cv::Mat measured_gradient_x;
  cv::Mat measured_gradient_y;
  cv::Sobel(measured_focused, measured_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(measured_focused, measured_gradient_y, CV_32F, 0, 1, 3, 0.125);
  cv::Mat measured_magnitude;
  cv::magnitude(measured_gradient_x, measured_gradient_y, measured_magnitude);

  cv::Scalar rendered_mean;
  cv::Scalar rendered_stddev;
  cv::Scalar measured_mean;
  cv::Scalar measured_stddev;
  cv::meanStdDev(rendered_magnitude, rendered_mean, rendered_stddev, alpha);
  cv::meanStdDev(measured_magnitude, measured_mean, measured_stddev, alpha);
  if (rendered_stddev[0] < 1.0e-3 || measured_stddev[0] < 1.0e-3) return;
  cv::Mat template_signal{
      (rendered_magnitude - rendered_mean[0]) / rendered_stddev[0]};
  cv::Mat measured_signal{
      (measured_magnitude - measured_mean[0]) / measured_stddev[0]};

  constexpr float ecc_scale{0.5f};
  cv::Mat template_signal_small;
  cv::Mat measured_signal_small;
  cv::Mat alpha_small;
  cv::resize(template_signal, template_signal_small, cv::Size(), ecc_scale,
             ecc_scale, cv::INTER_AREA);
  cv::resize(measured_signal, measured_signal_small, cv::Size(), ecc_scale,
             ecc_scale, cv::INTER_AREA);
  cv::resize(alpha, alpha_small, cv::Size(), ecc_scale, ecc_scale,
             cv::INTER_NEAREST);
  cv::Mat warp_matrix{cv::Mat::eye(2, 3, CV_32F)};
  try {
    model_ecc_correlation_ = float(cv::findTransformECC(
        template_signal_small, measured_signal_small, warp_matrix,
        cv::MOTION_AFFINE,
        cv::TermCriteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10,
                         1.0e-4},
        alpha_small, 3));
  } catch (const cv::Exception &) {
    model_ecc_correlation_ = 0.0f;
    return;
  }
  const float determinant{warp_matrix.at<float>(0, 0) *
                              warp_matrix.at<float>(1, 1) -
                          warp_matrix.at<float>(0, 1) *
                              warp_matrix.at<float>(1, 0)};
  if (!std::isfinite(model_ecc_correlation_) ||
      model_ecc_correlation_ < 0.2f || determinant < 0.7f ||
      determinant > 1.3f)
    return;
  warp_matrix.at<float>(0, 2) /= ecc_scale;
  warp_matrix.at<float>(1, 2) /= ecc_scale;

  struct Candidate {
    int u = 0;
    int v = 0;
    float score = -1.0f;
  };
  constexpr int cell_size{10};
  const int n_cells_u{(rendered_gray.cols + cell_size - 1) / cell_size};
  const int n_cells_v{(rendered_gray.rows + cell_size - 1) / cell_size};
  std::vector<Candidate> candidates(size_t(n_cells_u * n_cells_v));
  for (int v = 1; v < rendered_gray.rows - 1; ++v) {
    for (int u = 1; u < rendered_gray.cols - 1; ++u) {
      if (alpha.at<uchar>(v, u) == 0) continue;
      const float score{rendered_magnitude.at<float>(v, u)};
      if (score < 8.0f) continue;
      Candidate &candidate{
          candidates[size_t((v / cell_size) * n_cells_u + u / cell_size)]};
      if (score > candidate.score) candidate = Candidate{u, v, score};
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.score > b.score;
            });

  const float corner_u{textured_renderer_ptr_->corner_u()};
  const float corner_v{textured_renderer_ptr_->corner_v()};
  for (const auto &candidate : candidates) {
    if (candidate.score < 0.0f ||
        int(model_ecc_data_points_.size()) >= n_model_ecc_points_)
      break;
    const float correspondence_u{
        warp_matrix.at<float>(0, 0) * float(candidate.u) +
        warp_matrix.at<float>(0, 1) * float(candidate.v) +
        warp_matrix.at<float>(0, 2)};
    const float correspondence_v{
        warp_matrix.at<float>(1, 0) * float(candidate.u) +
        warp_matrix.at<float>(1, 1) * float(candidate.v) +
        warp_matrix.at<float>(1, 2)};
    const float displacement{
        std::hypot(correspondence_u - float(candidate.u),
                   correspondence_v - float(candidate.v))};
    if (displacement > 10.0f) continue;
    const ushort depth_value{
        rendered_depth.at<ushort>(candidate.v, candidate.u)};
    if (depth_value == std::numeric_limits<ushort>::max()) continue;
    const float center_u{corner_u + float(candidate.u) * inverse_scale};
    const float center_v{corner_v + float(candidate.v) * inverse_scale};
    const float depth{textured_renderer_ptr_->Depth(depth_value)};
    const Eigen::Vector3f center_f_camera{
        depth * (center_u - ppu_) / fu_, depth * (center_v - ppv_) / fv_,
        depth};
    model_ecc_data_points_.push_back(ModelEccDataPoint{
        camera2body_pose_ * center_f_camera,
        Eigen::Vector2f{corner_u + correspondence_u * inverse_scale,
                        corner_v + correspondence_v * inverse_scale}});
  }
}

float TextureModality::ModelTextureGradientCorrelation() {
  if (!textured_renderer_ptr_->StartRendering() ||
      !textured_renderer_ptr_->IsBodyVisible(body_ptr_->name()) ||
      !textured_renderer_ptr_->FetchColorImage())
    return -1.0f;
  const cv::Mat &rendered_color{
      textured_renderer_ptr_->focused_color_image()};
  cv::Mat rendered_gray;
  cv::cvtColor(rendered_color, rendered_gray, cv::COLOR_BGRA2GRAY);
  cv::Mat rendered_gradient_x;
  cv::Mat rendered_gradient_y;
  cv::Sobel(rendered_gray, rendered_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(rendered_gray, rendered_gradient_y, CV_32F, 0, 1, 3, 0.125);
  cv::Mat rendered_magnitude;
  cv::magnitude(rendered_gradient_x, rendered_gradient_y, rendered_magnitude);

  cv::Mat mask;
  cv::extractChannel(rendered_color, mask, 3);
  cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
  cv::erode(mask, mask,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{7, 7}));
  if (cv::countNonZero(mask) < 30) return -1.0f;

  cv::Mat measured_gray;
  cv::cvtColor(ObservedImage(), measured_gray, cv::COLOR_BGR2GRAY);
  cv::Mat measured_focused;
  const float inverse_scale{1.0f / textured_renderer_ptr_->scale()};
  const cv::Mat focus_mapping{
      (cv::Mat_<float>(2, 3) << inverse_scale, 0.0f,
       textured_renderer_ptr_->corner_u(), 0.0f, inverse_scale,
       textured_renderer_ptr_->corner_v())};
  cv::warpAffine(measured_gray, measured_focused, focus_mapping,
                 rendered_gray.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT);
  cv::Mat measured_gradient_x;
  cv::Mat measured_gradient_y;
  cv::Sobel(measured_focused, measured_gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(measured_focused, measured_gradient_y, CV_32F, 0, 1, 3, 0.125);
  cv::Mat measured_magnitude;
  cv::magnitude(measured_gradient_x, measured_gradient_y, measured_magnitude);

  cv::resize(rendered_magnitude, rendered_magnitude, cv::Size(), 0.5, 0.5,
             cv::INTER_AREA);
  cv::resize(measured_magnitude, measured_magnitude, cv::Size(), 0.5, 0.5,
             cv::INTER_AREA);
  cv::resize(mask, mask, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
  cv::Scalar rendered_mean;
  cv::Scalar rendered_stddev;
  cv::Scalar measured_mean;
  cv::Scalar measured_stddev;
  cv::meanStdDev(rendered_magnitude, rendered_mean, rendered_stddev, mask);
  cv::meanStdDev(measured_magnitude, measured_mean, measured_stddev, mask);
  const double denominator{rendered_stddev[0] * measured_stddev[0]};
  if (denominator < 1.0e-6) return -1.0f;
  const cv::Mat product{(rendered_magnitude - rendered_mean[0])
                            .mul(measured_magnitude - measured_mean[0])};
  return float(cv::mean(product, mask)[0] / denominator);
}

void TextureModality::ComputeModelHypothesis() {
  model_hypothesis_confidence_ = 0.0f;
  const Transform3fA original_pose{body_ptr_->body2world_pose()};
  model_hypothesis_target_pose_ = original_pose;
  constexpr float angle{4.0f * kPi / 180.0f};
  struct Hypothesis {
    Eigen::Vector3f axis{};
    float angle = 0.0f;
  };
  const std::array<Hypothesis, 7> hypotheses{
      Hypothesis{Eigen::Vector3f::UnitX(), 0.0f},
      Hypothesis{Eigen::Vector3f::UnitX(), angle},
      Hypothesis{Eigen::Vector3f::UnitX(), -angle},
      Hypothesis{Eigen::Vector3f::UnitY(), angle},
      Hypothesis{Eigen::Vector3f::UnitY(), -angle},
      Hypothesis{Eigen::Vector3f::UnitZ(), angle},
      Hypothesis{Eigen::Vector3f::UnitZ(), -angle}};
  std::array<float, 7> scores;
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    Transform3fA candidate_pose{original_pose};
    if (hypotheses[i].angle != 0.0f)
      candidate_pose.rotate(Eigen::AngleAxisf{hypotheses[i].angle,
                                              hypotheses[i].axis});
    body_ptr_->set_body2world_pose(candidate_pose);
    scores[i] = ModelTextureGradientCorrelation();
  }
  body_ptr_->set_body2world_pose(original_pose);

  size_t best_index{0};
  for (size_t i = 1; i < scores.size(); ++i) {
    if (scores[i] > scores[best_index]) best_index = i;
  }
  const float margin{scores[best_index] - scores[0]};
  if (best_index == 0 || scores[best_index] < 0.1f || margin < 0.015f)
    return;
  model_hypothesis_target_pose_ = original_pose;
  model_hypothesis_target_pose_.rotate(Eigen::AngleAxisf{
      hypotheses[best_index].angle, hypotheses[best_index].axis});
  const float margin_reliability{std::clamp(margin / 0.08f, 0.0f, 1.0f)};
  const float score_reliability{
      std::clamp((scores[best_index] - 0.05f) / 0.3f, 0.0f, 1.0f)};
  model_hypothesis_confidence_ =
      margin_reliability * score_reliability;
}

void TextureModality::ComputeVideoEccKeyframe() {
  video_ecc_keyframe_image_.release();
  video_ecc_keyframe_mask_.release();
  video_ecc_keyframe_body_points_.clear();
  video_ecc_keyframe_image_points_.clear();
  cv::Rect region_of_interest;
  float unused_scale;
  if (!CalculateScaleAndRegionOfInterest(&region_of_interest, &unused_scale))
    return;

  cv::Mat gray_image;
  cv::cvtColor(ObservedImage(), gray_image, cv::COLOR_BGR2GRAY);
  cv::resize(gray_image(region_of_interest), video_ecc_keyframe_image_,
             cv::Size{focused_image_size_, focused_image_size_});
  cv::equalizeHist(video_ecc_keyframe_image_, video_ecc_keyframe_image_);
  video_ecc_keyframe_mask_ =
      cv::Mat::zeros(focused_image_size_, focused_image_size_, CV_8UC1);
  const cv::Mat &silhouette_image{
      silhouette_renderer_ptr_->focused_silhouette_image()};
  const float width_scale{float(region_of_interest.width) /
                          float(focused_image_size_)};
  const float height_scale{float(region_of_interest.height) /
                           float(focused_image_size_)};
  for (int v = 0; v < focused_image_size_; ++v) {
    for (int u = 0; u < focused_image_size_; ++u) {
      const float image_u{region_of_interest.x + (float(u) + 0.5f) *
                                                     width_scale};
      const float image_v{region_of_interest.y + (float(v) + 0.5f) *
                                                     height_scale};
      const int silhouette_u{int(
          (image_u - silhouette_renderer_ptr_->corner_u()) *
              silhouette_renderer_ptr_->scale() +
          0.5f)};
      const int silhouette_v{int(
          (image_v - silhouette_renderer_ptr_->corner_v()) *
              silhouette_renderer_ptr_->scale() +
          0.5f)};
      if (silhouette_u >= 0 && silhouette_u < silhouette_image.cols &&
          silhouette_v >= 0 && silhouette_v < silhouette_image.rows &&
          silhouette_image.at<uchar>(silhouette_v, silhouette_u) ==
              body_ptr_->body_id())
        video_ecc_keyframe_mask_.at<uchar>(v, u) = 255;
    }
  }
  cv::erode(video_ecc_keyframe_mask_, video_ecc_keyframe_mask_,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{7, 7}));

  cv::Mat gradient_x;
  cv::Mat gradient_y;
  cv::Sobel(video_ecc_keyframe_image_, gradient_x, CV_32F, 1, 0, 3, 0.125);
  cv::Sobel(video_ecc_keyframe_image_, gradient_y, CV_32F, 0, 1, 3, 0.125);
  cv::Mat magnitude;
  cv::magnitude(gradient_x, gradient_y, magnitude);
  struct Candidate {
    int u = 0;
    int v = 0;
    float score = -1.0f;
  };
  constexpr int cell_size{10};
  const int n_cells{(focused_image_size_ + cell_size - 1) / cell_size};
  std::vector<Candidate> candidates(size_t(n_cells * n_cells));
  for (int v = 1; v < focused_image_size_ - 1; ++v) {
    for (int u = 1; u < focused_image_size_ - 1; ++u) {
      if (video_ecc_keyframe_mask_.at<uchar>(v, u) == 0) continue;
      const float score{magnitude.at<float>(v, u)};
      if (score < 8.0f) continue;
      Candidate &candidate{
          candidates[size_t((v / cell_size) * n_cells + u / cell_size)]};
      if (score > candidate.score) candidate = Candidate{u, v, score};
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              return a.score > b.score;
            });
  for (const auto &candidate : candidates) {
    if (candidate.score < 0.0f ||
        int(video_ecc_keyframe_body_points_.size()) >= n_video_ecc_points_)
      break;
    const cv::Point2f image_point{
        region_of_interest.x + (float(candidate.u) + 0.5f) * width_scale,
        region_of_interest.y + (float(candidate.v) + 0.5f) * height_scale};
    Eigen::Vector3f body_point;
    if (!Reconstruct3DPoint(image_point, &body_point)) continue;
    video_ecc_keyframe_body_points_.push_back(body_point);
    video_ecc_keyframe_image_points_.emplace_back(float(candidate.u),
                                                   float(candidate.v));
  }
}

void TextureModality::ComputeVideoEccCorrespondences() {
  video_ecc_data_points_.clear();
  video_ecc_correlation_ = 0.0f;
  if (video_ecc_keyframe_image_.empty() ||
      video_ecc_keyframe_body_points_.size() < 6)
    return;
  cv::Rect current_roi;
  float unused_scale;
  if (!CalculateScaleAndRegionOfInterest(&current_roi, &unused_scale)) return;
  cv::Mat current_gray;
  cv::cvtColor(ObservedImage(), current_gray, cv::COLOR_BGR2GRAY);
  cv::resize(current_gray(current_roi), current_gray,
             video_ecc_keyframe_image_.size());
  cv::equalizeHist(current_gray, current_gray);

  cv::Mat keyframe_small;
  cv::Mat current_small;
  cv::Mat mask_small;
  cv::resize(video_ecc_keyframe_image_, keyframe_small, cv::Size(), 0.5, 0.5,
             cv::INTER_AREA);
  cv::resize(current_gray, current_small, cv::Size(), 0.5, 0.5,
             cv::INTER_AREA);
  cv::resize(video_ecc_keyframe_mask_, mask_small, cv::Size(), 0.5, 0.5,
             cv::INTER_NEAREST);
  keyframe_small.convertTo(keyframe_small, CV_32F, 1.0 / 255.0);
  current_small.convertTo(current_small, CV_32F, 1.0 / 255.0);
  cv::Mat warp_matrix{cv::Mat::eye(2, 3, CV_32F)};
  try {
    video_ecc_correlation_ = float(cv::findTransformECC(
        keyframe_small, current_small, warp_matrix, cv::MOTION_AFFINE,
        cv::TermCriteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10,
                         1.0e-4},
        mask_small, 3));
  } catch (const cv::Exception &) {
    return;
  }
  const float determinant{warp_matrix.at<float>(0, 0) *
                              warp_matrix.at<float>(1, 1) -
                          warp_matrix.at<float>(0, 1) *
                              warp_matrix.at<float>(1, 0)};
  if (!std::isfinite(video_ecc_correlation_) ||
      video_ecc_correlation_ < 0.4f || determinant < 0.7f ||
      determinant > 1.3f)
    return;
  warp_matrix.at<float>(0, 2) *= 2.0f;
  warp_matrix.at<float>(1, 2) *= 2.0f;

  const float width_scale{float(current_roi.width) /
                          float(focused_image_size_)};
  const float height_scale{float(current_roi.height) /
                           float(focused_image_size_)};
  for (size_t i = 0; i < video_ecc_keyframe_body_points_.size(); ++i) {
    const auto &point{video_ecc_keyframe_image_points_[i]};
    const float correspondence_u{warp_matrix.at<float>(0, 0) * point.x +
                                 warp_matrix.at<float>(0, 1) * point.y +
                                 warp_matrix.at<float>(0, 2)};
    const float correspondence_v{warp_matrix.at<float>(1, 0) * point.x +
                                 warp_matrix.at<float>(1, 1) * point.y +
                                 warp_matrix.at<float>(1, 2)};
    if (correspondence_u < 0.0f ||
        correspondence_u >= float(focused_image_size_) ||
        correspondence_v < 0.0f ||
        correspondence_v >= float(focused_image_size_))
      continue;
    video_ecc_data_points_.push_back(VideoEccDataPoint{
        video_ecc_keyframe_body_points_[i],
        Eigen::Vector2f{current_roi.x +
                            (correspondence_u + 0.5f) * width_scale,
                        current_roi.y +
                            (correspondence_v + 0.5f) * height_scale}});
  }
}

bool TextureModality::CalculateScaleAndRegionOfInterest(
    cv::Rect *region_of_interest, float *scale) const {
  // Project sphere into image
  float r = 0.5f * body_ptr_->maximum_body_diameter();
  auto translation{body2camera_pose_.translation()};
  float x = translation(0);
  float y = translation(1);
  float z = translation(2);
  if (z < r * 1.5f) return false;
  float abs_x = std::abs(x);
  float abs_y = std::abs(y);
  float x2 = x * x;
  float y2 = y * y;
  float z2 = z * z;
  float r2 = r * r;
  float rz = r * z;
  float z2_r2 = z2 - r2;
  float z3_zr2 = z2_r2 * z;
  float r_u = fu_ * (abs_x * r2 + rz * sqrtf(z2_r2 + x2)) / z3_zr2;
  float r_v = fv_ * (abs_y * r2 + rz * sqrtf(z2_r2 + y2)) / z3_zr2;
  float center_u = x * fu_ / z + ppu_;
  float center_v = y * fv_ / z + ppv_;

  // Calculate region of interest
  int u_min = int(center_u - r_u - kRegionOfInterestMargin + 0.5f);
  int u_max = int(center_u + r_u + kRegionOfInterestMargin + 0.5f);
  int v_min = int(center_v - r_v - kRegionOfInterestMargin + 0.5f);
  int v_max = int(center_v + r_v + kRegionOfInterestMargin + 0.5f);
  u_min = std::max(u_min, 0);
  u_max = std::min(u_max, image_width_minus_1_);
  v_min = std::max(v_min, 0);
  v_max = std::min(v_max, image_height_minus_1_);
  if (u_min >= u_max || v_min >= v_max) return false;
  region_of_interest->x = u_min;
  region_of_interest->y = v_min;
  region_of_interest->width = u_max - u_min;
  region_of_interest->height = v_max - v_min;

  // Calculate scale
  *scale = float(focused_image_size_) / std::max(2.0f * r_u, 2.0f * r_v);
  return true;
}

void TextureModality::ComputeKeyframeData() {
  if (points_keyframes_.size() >= n_keyframes_) {
    points_keyframes_.pop_front();
    if (use_optical_flow_tracking_) {
      image_points_keyframes_.pop_front();
      gray_images_keyframes_.pop_front();
    }
#ifdef USE_CUDA
    if (descriptor_type_ == DescriptorType::ORB_CUDA)
      descriptors_keyframes_cuda_.pop_front();
    else
#endif
      descriptors_keyframes_.pop_front();
  }

  // Fetch depth and silhouette images
  if (!silhouette_renderer_ptr_->IsBodyVisible(body_ptr_->name())) return;
  silhouette_renderer_ptr_->FetchDepthImage();
  silhouette_renderer_ptr_->FetchSilhouetteImage();
  bool body_visible_depth;
  if (model_occlusions_) {
    body_visible_depth = depth_renderer_ptr_->IsBodyVisible(body_ptr_->name());
    if (body_visible_depth) depth_renderer_ptr_->FetchDepthImage();
  }

  // Reconstruct points
  std::vector<size_t> indexes;
  std::vector<Eigen::Vector3f> points_keyframe;
  for (size_t i = 0; i < keypoints_.size(); ++i) {
    Eigen::Vector3f point;
    if (!Reconstruct3DPoint(keypoints_[i].pt, &point)) continue;
    if (!IsPointValid(point, measure_occlusions_,
                      model_occlusions_ && body_visible_depth))
      continue;
    points_keyframe.push_back(std::move(point));
    indexes.push_back(i);
  }
  points_keyframes_.push_back(std::move(points_keyframe));

  if (use_optical_flow_tracking_) {
    std::vector<cv::Point2f> image_points_keyframe;
    image_points_keyframe.reserve(indexes.size());
    for (size_t index : indexes)
      image_points_keyframe.push_back(keypoints_[index].pt);
    image_points_keyframes_.push_back(std::move(image_points_keyframe));
    cv::Mat gray_image;
    cv::cvtColor(ObservedImage(), gray_image, cv::COLOR_BGR2GRAY);
    gray_images_keyframes_.push_back(std::move(gray_image));
  }

  // Copy descriptors
#ifdef USE_CUDA
  if (descriptor_type_ == DescriptorType::ORB_CUDA) {
    cv::cuda::GpuMat descriptors_keyframe_cuda(
        indexes.size(), descriptors_cuda_.cols, descriptors_cuda_.type());
    for (size_t i = 0; i < indexes.size(); ++i)
      descriptors_cuda_.row(indexes[i])
          .copyTo(descriptors_keyframe_cuda.row(i));
    descriptors_keyframes_cuda_.push_back(std::move(descriptors_keyframe_cuda));
  } else
#endif
  {
    cv::Mat descriptors_keyframe(int(indexes.size()), descriptors_.cols,
                                 descriptors_.type());
    for (size_t i = 0; i < indexes.size(); ++i)
      descriptors_.row(int(indexes[i]))
          .copyTo(descriptors_keyframe.row(int(i)));
    descriptors_keyframes_.push_back(std::move(descriptors_keyframe));
  }

  if (use_video_ecc_tracking_) ComputeVideoEccKeyframe();

  // Store orientation
  orientation_last_keyframe_ =
      body2camera_pose_.rotation().inverse() *
      body2camera_pose_.translation().matrix().normalized();
  keyframe_age_ = 0;
}

bool TextureModality::Reconstruct3DPoint(const cv::Point2f &center,
                                         Eigen::Vector3f *center_f_body) const {
  const cv::Mat &silhouette_image{
      silhouette_renderer_ptr_->focused_silhouette_image()};
  const cv::Mat &depth_image{silhouette_renderer_ptr_->focused_depth_image()};
  uchar body_id = body_ptr_->body_id();

  // Compute coordinates for silhouette renderer
  int u_silhouette = int((center.x - silhouette_renderer_ptr_->corner_u()) *
                             silhouette_renderer_ptr_->scale() +
                         0.5f);
  int v_silhouette = int((center.y - silhouette_renderer_ptr_->corner_v()) *
                             silhouette_renderer_ptr_->scale() +
                         0.5f);
  if (u_silhouette < 0 || u_silhouette > silhouette_image_size_minus_1_ ||
      v_silhouette < 0 || v_silhouette > silhouette_image_size_minus_1_)
    return false;

  // Check if point is on body
  if (silhouette_image.at<uchar>(v_silhouette, u_silhouette) != body_id)
    return false;

  // Calculate 3D model point
  ushort depth_image_value = depth_image.at<ushort>(v_silhouette, u_silhouette);
  float depth = silhouette_renderer_ptr_->Depth(depth_image_value);
  Eigen::Vector3f center_f_camera{depth * (center.x - ppu_) / fu_,
                                  depth * (center.y - ppv_) / fv_, depth};
  *center_f_body = camera2body_pose_ * center_f_camera;
  return true;
}

bool TextureModality::IsPointValid(const Eigen::Vector3f &center_f_body,
                                   bool measure_occlusions,
                                   bool model_occlusions) {
  if (measure_occlusions)
    if (!IsPointUnoccludedMeasured(center_f_body)) return false;
  if (model_occlusions)
    if (!IsPointUnoccludedModeled(center_f_body)) return false;
  return true;
}

bool TextureModality::IsPointUnoccludedMeasured(
    const Eigen::Vector3f &center_f_body) const {
  // Calculate center on depth image
  Eigen::Vector3f center_f_depth_camera{body2depth_camera_pose_ *
                                        center_f_body};
  float center_u =
      center_f_depth_camera(0) * depth_fu_ / center_f_depth_camera(2) +
      depth_ppu_;
  float center_v =
      center_f_depth_camera(1) * depth_fv_ / center_f_depth_camera(2) +
      depth_ppv_;

  // Precalculate variables in pixel coordinates
  float meter_to_pixel = depth_fu_ / center_f_depth_camera(2);
  float diameter = 2.0f * measured_occlusion_radius_ * meter_to_pixel;
  int stride = int(diameter / kMaxNOcclusionStrides + 1.0f);
  int n_strides = int(diameter / stride + 0.5f);
  int rounded_diameter = n_strides * stride;
  float rounded_radius = 0.5f * float(rounded_diameter);

  // Calculate limits for iteration
  int u_min = int(center_u - rounded_radius + 0.5f);
  int v_min = int(center_v - rounded_radius + 0.5f);
  int u_max = u_min + rounded_diameter;
  int v_max = v_min + rounded_diameter;
  u_min = std::max(u_min, 0);
  v_min = std::max(v_min, 0);
  u_max = std::min(u_max, depth_image_width_minus_1_);
  v_max = std::min(v_max, depth_image_height_minus_1_);

  // Iterate over pixels to check if depth value is big enough
  ushort depth;
  ushort min_depth =
      ushort((center_f_depth_camera(2) - measured_occlusion_threshold_) /
             depth_scale_);
  int u, v;
  const cv::Mat &image{depth_camera_ptr_->image()};
  const ushort *ptr_image;
  for (v = v_min; v <= v_max; v += stride) {
    ptr_image = image.ptr<ushort>(v);
    for (u = u_min; u <= u_max; u += stride) {
      depth = ptr_image[u];
      if (depth > 0 && depth < min_depth) return false;
    }
  }
  return true;
}

bool TextureModality::IsPointUnoccludedModeled(
    const Eigen::Vector3f &center_f_body) const {
  Eigen::Vector3f center_f_camera{body2camera_pose_ * center_f_body};

  // Precalculate variables in pixel coordinates of focused image
  float meter_to_pixel =
      (fu_ / center_f_camera(2)) * depth_renderer_ptr_->scale();
  float diameter = 2.0f * modeled_occlusion_radius_ * meter_to_pixel;
  int stride = int(diameter / kMaxNOcclusionStrides + 1.0f);
  int n_strides = int(diameter / stride + 0.5f);
  int rounded_diameter = n_strides * stride;
  float rounded_radius = 0.5f * float(rounded_diameter);

  // Calculate limits for iteration in focused image
  float center_u = center_f_camera(0) * fu_ / center_f_camera(2) + ppu_;
  float center_v = center_f_camera(1) * fv_ / center_f_camera(2) + ppv_;
  float focused_center_u = (center_u - depth_renderer_ptr_->corner_u()) *
                           depth_renderer_ptr_->scale();
  float focused_center_v = (center_v - depth_renderer_ptr_->corner_v()) *
                           depth_renderer_ptr_->scale();
  int u_min = int(focused_center_u - rounded_radius + 0.5f);
  int v_min = int(focused_center_v - rounded_radius + 0.5f);
  int u_max = u_min + rounded_diameter;
  int v_max = v_min + rounded_diameter;
  u_min = std::max(u_min, 0);
  v_min = std::max(v_min, 0);
  u_max = std::min(u_max, depth_image_size_minus_1_);
  v_max = std::min(v_max, depth_image_size_minus_1_);

  // Find minimum depth value
  ushort min_depth_value = std::numeric_limits<ushort>::max();
  int u, v;
  const cv::Mat &image{depth_renderer_ptr_->focused_depth_image()};
  const ushort *ptr_image;
  for (v = v_min; v <= v_max; v += stride) {
    ptr_image = image.ptr<ushort>(v);
    for (u = u_min; u <= u_max; u += stride) {
      min_depth_value = std::min(min_depth_value, ptr_image[u]);
    }
  }

  float min_depth = depth_renderer_ptr_->Depth(min_depth_value);
  float min_allowed_depth = center_f_camera(2) - modeled_occlusion_threshold_;
  return min_depth > min_allowed_depth;
}

void TextureModality::ShowAndSaveImage(const std::string &title, int save_index,
                                       const cv::Mat &image) const {
  if (display_visualization_) cv::imshow(title, image);
  if (save_visualizations_) {
    std::filesystem::path path{
        save_directory_ /
        (title + "_" + std::to_string(save_index) + "." + save_image_type_)};
    cv::imwrite(path.string(), image);
  }
}

void TextureModality::VisualizePointsColorImage(const std::string &title,
                                                int save_index) const {
  cv::Mat visualization_image;
  color_camera_ptr_->image().copyTo(visualization_image);
  DrawPoints(cv::Vec3b{24, 184, 234}, &visualization_image);
  ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
}

void TextureModality::VisualizePointsDepthImage(const std::string &title,
                                                int save_index) const {
  cv::Mat visualization_image;
  cv::cvtColor(depth_camera_ptr_->NormalizedDepthImage(
                   visualization_min_depth_, visualization_max_depth_),
               visualization_image, cv::COLOR_GRAY2BGR);
  DrawDepthPoints(cv::Vec3b{187, 117, 0}, &visualization_image);
  ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
}

void TextureModality::VisualizePointsDepthRendering(const std::string &title,
                                                    int save_index) const {
  cv::Mat visualization_image;
  depth_renderer_ptr_->FetchDepthImage();
  cv::cvtColor(depth_renderer_ptr_->NormalizedFocusedDepthImage(
                   visualization_min_depth_, visualization_max_depth_),
               visualization_image, cv::COLOR_GRAY2BGR);
  DrawFocusedPoints(depth_renderer_ptr_, cv::Vec3b{24, 184, 234},
                    &visualization_image);
  ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
}

void TextureModality::VisualizePointsSilhouetteRendering(
    const std::string &title, int save_index) const {
  cv::Mat visualization_image;
  silhouette_renderer_ptr_->FetchDepthImage();
  cv::cvtColor(silhouette_renderer_ptr_->focused_silhouette_image(),
               visualization_image, cv::COLOR_GRAY2BGR);
  DrawFocusedPoints(silhouette_renderer_ptr_, cv::Vec3b{24, 184, 234},
                    &visualization_image);
  ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
}

void TextureModality::VisualizeCorrespondences(const std::string &title,
                                               int save_index) const {
  cv::Mat visualization_image;
  color_camera_ptr_->image().copyTo(visualization_image);
  DrawCorrespondences(cv::Vec3b{187, 117, 0}, cv::Vec3b{61, 63, 179},
                      cv::Vec3b{24, 184, 234}, &visualization_image);
  ShowAndSaveImage(name_ + "_" + title, save_index, visualization_image);
}

void TextureModality::DrawPoints(const cv::Vec3b &color_point,
                                 cv::Mat *image) const {
  for (const auto &point : points_keyframes_.back()) {
    DrawPointInImage(body2camera_pose_ * point, color_point,
                     color_camera_ptr_->intrinsics(), image);
  }
}

void TextureModality::DrawDepthPoints(const cv::Vec3b &color_point,
                                      cv::Mat *image) const {
  for (const auto &point : points_keyframes_.back()) {
    DrawPointInImage(body2depth_camera_pose_ * point, color_point,
                     depth_camera_ptr_->intrinsics(), image);
  }
}

void TextureModality::DrawFocusedPoints(
    const std::shared_ptr<FocusedRenderer> &renderer_ptr,
    const cv::Vec3b &color_point, cv::Mat *image) const {
  for (const auto &point : points_keyframes_.back()) {
    DrawFocusedPointInImage(body2camera_pose_ * point, color_point,
                            renderer_ptr->intrinsics(),
                            renderer_ptr->corner_u(), renderer_ptr->corner_v(),
                            renderer_ptr->scale(), image);
  }
}

void TextureModality::DrawCorrespondences(
    const cv::Vec3b &color_point, const cv::Vec3b &color_correspondence_point,
    const cv::Vec3b &color_correspondence, cv::Mat *image) const {
  for (const auto &data_point : data_points_) {
    cv::Point2i center{int(data_point.center(0)), int(data_point.center(1))};
    cv::Point2i correspondence_center{int(data_point.correspondence_center(0)),
                                      int(data_point.correspondence_center(1))};
    cv::line(*image, center, correspondence_center, color_correspondence);
    cv::circle(*image, center, 1, color_point, cv::FILLED);
    cv::circle(*image, correspondence_center, 1, color_correspondence_point,
               cv::FILLED);
  }
}

float TextureModality::TukeyNorm(float error) {
  if (std::abs(error) <= tukey_norm_constant_)
    return powf(tukey_norm_constant_, 2.0f) / 6.0f *
           (1.0f - powf(1.0f - powf(error / tukey_norm_constant_, 2.0f), 3.0f));
  else
    return powf(tukey_norm_constant_, 2.0f) / 6.0f;
}

bool TextureModality::IsSetup() const {
  if (!set_up_) {
    std::cerr << "Set up texture modality " << name_ << " first" << std::endl;
    return false;
  }
  return true;
}

}  // namespace m3t
