// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber and Mariam Elsayed,
// German Aerospace Center (DLR)

#ifndef M3T_INCLUDE_M3T_TEXTURE_MODALITY_H_
#define M3T_INCLUDE_M3T_TEXTURE_MODALITY_H_

#include <filesystem/filesystem.h>
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/common.h>
#ifdef USE_ONNXRUNTIME
#include <m3t/disk_feature_extractor.h>
#endif
#include <m3t/modality.h>
#include <m3t/renderer.h>
#include <m3t/silhouette_renderer.h>
#include <m3t/textured_renderer.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#ifdef USE_CUDA
#include <opencv2/cudafeatures2d.hpp>
#endif
#include <deque>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>

namespace m3t {

/**
 * \brief \ref Modality class that implements a texture-based modality, which
 * uses information from a \ref ColorCamera to calculate the gradient vector and
 * Hessian matrix that are used by a \ref Link and \ref Optimizer to update the
 * \ref Body pose.
 *
 * \details The modality required a \ref FocusedSilhouetteRenderer that is used
 * to reconstruct 3D feature points for keyframes. The modality is able to
 * measure occlusions using images from a referenced \ref DepthCamera that is
 * close to the referenced \ref ColorCamera and model occlusions using
 * renderings from a \ref FocusedDepthRenderer object.
 *
 * @param color_camera_ptr referenced \ref ColorCamera from which images are
 * taken.
 * @param silhouette_renderer_ptr referenced \ref FocusedSilhouetteRenderer that
 * is used to reconstruct 3D feature points for keyframes. It has to be
 * configured with the IDType BODY.
 * @param depth_camera_ptr referenced \ref DepthCamera that is used to for
 * measured occlusion handling.
 * @param depth_renderer_ptr referenced \ref FocusedDepthRenderer that is used
 * for modeled occlusion handling.
 * @param descriptor_type specifies the \ref DescriptorType that defines which
 * descriptor and detector pair is used.
 * @param focused_image_size specifies the size of the image crop in which
 * features are detected.
 * @param descriptor_distance_threshold specifies the minimum difference of the
 * Hamming distance between the best and second-best match.
 * @param tukey_norm_constant defines the maximum expected value for valid
 * residual errors in image space in pixel.
 * @param standard_deviations user-defined standard deviation for each
 * iteration in pixels. If fewer values than `n_corr_iterations` are given, the
 * last provided standard deviation is used.
 * @param max_keyframe_rotation_difference defines after which rotational
 * difference in radian a new keyframe is reconstructed.
 * @param max_keyframe_age specifies after how many iterations a new keyframe
 * has to be generated, even if the rotational difference criterion is not
 * fulfilled.
 * @param n_keyframes how many keyframes are considered at the same time.
 * @param orb_n_features number of features considered by ORB detector.
 * @param orb_scale_factor scale factor used by ORB detector.
 * @param orb_n_levels number of levels considered by ORB detector.
 * @param brisk_threshold threshold used by BRISK detector.
 * @param brisk_octave number of detection octaves considered by BRISK detector.
 * @param brisk_pattern_scale scale applied to the pattern that is used to
 * sample the neighborhood of a BRISK keypoint.
 * @param daisy_radius radius of the DAISY descriptor at initial scale.
 * @param daisy_q_radius radial range division quantity for DAISY descriptor.
 * @param daisy_q_theta angular range division quantity for DAISY descriptor.
 * @param daisy_q_hist gradient orientations range division quantity for DAISY
 * descriptor.
 * @param freak_orientation_normalized enable orientation normalization for
 * FREAK descriptor.
 * @param freak_scale_normalized enable scale normalization for FREAK
 * descriptor.
 * @param freak_pattern_scale scaling of the FREAK description pattern.
 * @param freak_n_octaves number of octaves covered by detected keypoints for
 * FREAK descriptor.
 * @param sift_n_features number of best features to retain by SIFT detector.
 * @param sift_n_octave_layers  number of layers in each octave considered by
 * SIFT detector
 * @param sift_contrast_threshold contrast threshold used to filter out weak
 * features in semi-uniform regions by SIFT detector.
 * @param sift_edge_threshold threshold used to filter out edge-like features by
 * SIFT detector.
 * @param sift_sigma sigma of the Gaussian applied to the input image at the
 * octave #0 by SIFT detector.
 * @param measured_occlusion_radius radius in meter that defines the area in
 * which depth measurements from a \ref DepthCamera are considered for occlusion
 * handling.
 * @param measured_occlusion_threshold defines how much smaller the minimum
 * value from the \ref DepthCamera is allowed to be compared to the value that
 * is expected from the current estimate.
 * @param modeled_occlusion_radius radius in meter that defines the area in
 * which depth measurements from a \ref FocusedDepthRenderer are considered for
 * occlusion handling.
 * @param modeled_occlusion_threshold defines how much smaller the minimum value
 * from the \ref FocusedDepthRenderer is allowed to be compared to the value
 * that is expected from the current estimate.
 * @param visualize_correspondences_correspondence visualize correspondence
 * point pairs during `VisualizeCorrespondences()`.
 * @param visualize_correspondences_optimization visualize keyframe points
 * during `VisualizeOptimization()`.
 * @param visualize_points_result visualize keyframe points during
 * `VisualizeResults()`.
 * @param visualize_points_depth_image_result visualize keyframe points on the
 * depth image, which is used for occlusion handling, during
 * `VisualizeResults()`.
 * @param visualize_points_silhouette_rendering_result  visualize keyframe
 * points on the silhouette rendering, which is used for reconstruction, during
 * `VisualizeResults()`.
 * @param visualize_points_depth_rendering_result  visualize keyframe points on
 * the depth rendering, which is used for occlusion handling, during
 * `VisualizeResults()`.
 * @param visualization_min_depth minimum depth for normalization of depth
 * images in visualization.
 * @param visualization_max_depth maximum depth for normalization of depth
 * images in visualization.
 */
class TextureModality : public Modality {
 private:
  static constexpr int kRegionOfInterestMargin = 10;  // pixels
  static constexpr int kMaxNOcclusionStrides = 5;

  // Data for correspondence point calculated during CalculateCorrespondences
  struct DataPoint {
    Eigen::Vector3f center_f_body{};
    Eigen::Vector3f center_f_camera{};
    Eigen::Vector2f center;
    Eigen::Vector2f correspondence_center;
    bool from_model_texture = false;
    bool from_model_texture_disk = false;
  };

  struct PhotometricDataPoint {
    Eigen::Vector3f center_f_body{};
    float model_intensity = 0.0f;
  };

  struct ModelContourDataPoint {
    Eigen::Vector3f center_f_body{};
    Eigen::Vector2f normal{};
    Eigen::Vector2f correspondence_center{};
  };

  struct ModelEccDataPoint {
    Eigen::Vector3f center_f_body{};
    Eigen::Vector2f correspondence_center{};
  };

  struct VideoEccDataPoint {
    Eigen::Vector3f center_f_body{};
    Eigen::Vector2f correspondence_center{};
  };

 public:
  /**
   * \brief Enum that defines considered descriptor and detector pairs.
   * @param BRISK uses the BRISK descriptor and BRISK detector.
   * @param DAISY uses the DAISY descriptor and ORB detector.
   * @param FREAK uses the FREAK descriptor and ORB detector.
   * @param SIFT uses the SIFT descriptor and SIFT detector.
   * @param ORB uses the ORB descriptor and ORB detector.
   * @param ORB_CUDA uses the ORB descriptor and ORB detector with CUDA support.
   */
  enum class DescriptorType {
    BRISK = 0,
    DAISY = 1,
    FREAK = 2,
    SIFT = 3,
    ORB = 4,
#ifdef USE_CUDA
    ORB_CUDA = 5
#endif
  };

  // Constructors and setup methods
  TextureModality(const std::string &name,
                  const std::shared_ptr<Body> &body_ptr,
                  const std::shared_ptr<ColorCamera> &color_camera_ptr,
                  const std::shared_ptr<FocusedSilhouetteRenderer>
                      &silhouette_renderer_ptr);
  TextureModality(const std::string &name,
                  const std::filesystem::path &metafile_path,
                  const std::shared_ptr<Body> &body_ptr,
                  const std::shared_ptr<ColorCamera> &color_camera_ptr,
                  const std::shared_ptr<FocusedSilhouetteRenderer>
                      &silhouette_renderer_ptr);
  bool SetUp() override;

  // Setters data
  void set_color_camera_ptr(
      const std::shared_ptr<ColorCamera> &color_camera_ptr);
  void set_silhouette_renderer_ptr(
      const std::shared_ptr<FocusedSilhouetteRenderer>
          &silhouette_renderer_ptr);

  // Setters for general parameters
  void set_descriptor_type(DescriptorType descriptor_type);
  void set_focused_image_size(int focused_image_size);
  void set_use_optical_flow_tracking(bool use_optical_flow_tracking);
  void set_use_geometry_guided_matching(bool use_geometry_guided_matching);
  void set_use_model_texture_sift(bool use_model_texture_sift);
  void set_use_model_texture_akaze(bool use_model_texture_akaze);
  void set_use_model_texture_disk(
      bool use_model_texture_disk,
      const std::filesystem::path &model_path = {});
  void set_use_adaptive_model_texture_disk(
      bool use_adaptive_model_texture_disk,
      const std::filesystem::path &model_path = {});
  void set_use_model_texture_disk_fusion(
      bool use_model_texture_disk_fusion,
      const std::filesystem::path &model_path = {});
  void set_use_video_ecc_tracking(bool use_video_ecc_tracking);
  void UseModelTextureMatching(
      const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr);
  void DoNotUseModelTextureMatching();
  void UseModelPhotometricTracking(
      const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr);
  void DoNotUseModelPhotometricTracking();
  void UseModelContourTracking(
      const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr);
  void DoNotUseModelContourTracking();
  void UseModelEccTracking(
      const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr);
  void DoNotUseModelEccTracking();
  void UseModelHypothesisSearch(
      const std::shared_ptr<FocusedTexturedRenderer> &textured_renderer_ptr);
  void DoNotUseModelHypothesisSearch();
  void set_model_texture_debug_directory(
      const std::filesystem::path &debug_directory);
  void set_model_texture_disk_ransac_threshold(float threshold);
  void set_model_texture_disk_input_size(int input_size);
  void set_use_adaptive_model_texture_geometry(
      bool use_adaptive_model_texture_geometry);
  void set_use_model_texture_pnp_geometry(bool use_model_texture_pnp_geometry);
  void set_descriptor_distance_threshold(float descriptor_distance_threshold);
  void set_tukey_norm_constant(float tukey_norm_constant);
  void set_standard_deviations(const std::vector<float> &standard_deviations);
  void set_max_keyframe_rotation_difference(
      float max_keyframe_rotation_difference);
  void set_max_keyframe_age(int max_keyframe_age);
  void set_n_keyframes(int n_keyframes);

  // Setters for feature detection
  void set_orb_n_features(int orb_n_features);
  void set_orb_scale_factor(float orb_scale_factor);
  void set_orb_n_levels(int orb_n_levels);
  void set_brisk_threshold(int brisk_threshold);
  void set_brisk_octave(int brisk_octave);
  void set_brisk_pattern_scale(float brisk_pattern_scale);
  void set_daisy_radius(float daisy_radius);
  void set_daisy_q_radius(int daisy_q_radius);
  void set_daisy_q_theta(int daisy_q_theta);
  void set_daisy_q_hist(int daisy_q_hist);
  void set_freak_orientation_normalized(bool freak_orientation_normalized);
  void set_freak_scale_normalized(bool freak_scale_normalized);
  void set_freak_pattern_scale(float freak_pattern_scale);
  void set_freak_n_octaves(int freak_n_octaves);
  void set_sift_n_features(int sift_n_features);
  void set_sift_n_octave_layers(int sift_n_octave_layers);
  void set_sift_contrast_threshold(double sift_contrast_threshold);
  void set_sift_edge_threshold(double sift_edge_threshold);
  void set_sift_sigma(double sift_sigma);

  // Setters for occlusion handling
  void MeasureOcclusions(const std::shared_ptr<DepthCamera> &depth_camera_ptr);
  void DoNotMeasureOcclusions();
  void ModelOcclusions(
      const std::shared_ptr<FocusedDepthRenderer> &depth_renderer_ptr);
  void DoNotModelOcclusions();
  void set_measured_occlusion_radius(float measured_occlusion_radius);
  void set_measured_occlusion_threshold(float measured_occlusion_threshold);
  void set_modeled_occlusion_radius(float modeled_occlusion_radius);
  void set_modeled_occlusion_threshold(float modeled_occlusion_threshold);

  // Setters to turn on individual visualizations
  void set_visualize_correspondences_correspondence(
      bool visualize_correspondences_correspondence);
  void set_visualize_corresopndences_optimization(
      bool visualize_correspondences_optimization);
  void set_visualize_points_result(bool visualize_points_result);
  void set_visualize_points_depth_image_result(
      bool visualize_points_depth_image_result);
  void set_visualize_points_silhouette_rendering_result(
      bool visualize_points_silhouette_rendering_result);
  void set_visualize_points_depth_rendering_result(
      bool visualize_points_depth_rendering_result);

  // Setters for visualization parameters
  void set_visualization_min_depth(float visualization_min_depth);
  void set_visualization_max_depth(float visualization_max_depth);
  void set_collect_diagnostics(bool collect_diagnostics);

  // Main methods
  bool StartModality(int iteration, int corr_iteration) override;
  bool CalculateCorrespondences(int iteration, int corr_iteration) override;
  bool VisualizeCorrespondences(int save_idx) override;
  bool CalculateGradientAndHessian(int iteration, int corr_iteration,
                                   int opt_iteration) override;
  bool VisualizeOptimization(int save_idx) override;
  bool CalculateResults(int iteration) override;
  bool VisualizeResults(int save_idx) override;

  // Getters data
  const std::shared_ptr<ColorCamera> &color_camera_ptr() const;
  const std::shared_ptr<DepthCamera> &depth_camera_ptr() const;
  const std::shared_ptr<FocusedSilhouetteRenderer> &silhouette_renderer_ptr()
      const;
  const std::shared_ptr<FocusedDepthRenderer> &depth_renderer_ptr() const;
  std::shared_ptr<Model> model_ptr() const override;
  std::vector<std::shared_ptr<Camera>> camera_ptrs() const override;
  std::vector<std::shared_ptr<Renderer>> start_modality_renderer_ptrs()
      const override;
  std::vector<std::shared_ptr<Renderer>> results_renderer_ptrs() const override;

  // Getters for general parameters
  DescriptorType descriptor_type() const;
  int focused_image_size() const;
  bool use_optical_flow_tracking() const;
  bool use_geometry_guided_matching() const;
  bool use_model_texture_matching() const;
  bool use_model_photometric_tracking() const;
  bool use_model_contour_tracking() const;
  bool use_model_ecc_tracking() const;
  bool use_model_hypothesis_search() const;
  bool use_video_ecc_tracking() const;
  float descriptor_distance_threshold() const;
  float tukey_norm_constant() const;
  const std::vector<float> &standard_deviations() const;
  float max_keyframe_rotation_difference() const;
  int max_keyframe_age() const;
  int n_keyframes() const;

  // Getters for feature detection
  int orb_n_features() const;
  float orb_scale_factor() const;
  int orb_n_levels() const;
  int brisk_threshold() const;
  int brisk_octave() const;
  float brisk_pattern_scale() const;
  float daisy_radius() const;
  int daisy_q_radius() const;
  int daisy_q_theta() const;
  int daisy_q_hist() const;
  bool freak_orientation_normalized() const;
  bool freak_scale_normalized() const;
  float freak_pattern_scale() const;
  int freak_n_octaves() const;
  int sift_n_features() const;
  int sift_n_octave_layers() const;
  double sift_contrast_threshold() const;
  double sift_edge_threshold() const;
  double sift_sigma() const;

  // Getters for occlusion handling
  bool measure_occlusions() const;
  float measured_occlusion_radius() const;
  float measured_occlusion_threshold() const;
  bool model_occlusions() const;
  float modeled_occlusion_radius() const;
  float modeled_occlusion_threshold() const;
  int n_unoccluded_iterations() const;
  int min_n_unoccluded_points() const;

  // Getters to turn on individual visualizations
  bool visualize_correspondences_correspondence() const;
  bool visualize_correspondences_optimization() const;
  bool visualize_points_result() const;
  bool visualize_points_depth_image_result() const;
  bool visualize_points_silhouette_rendering_result() const;
  bool visualize_points_depth_rendering_result() const;

  // Getters for visualization parameters
  float visualization_min_depth() const;
  float visualization_max_depth() const;

  // Getters for lightweight correspondence diagnostics
  size_t n_correspondence_points() const;
  int n_valid_gradient_points() const;
  float mean_reprojection_error() const;
  float inlier_fraction() const;
  size_t n_model_texture_correspondence_points() const;
  int n_valid_model_texture_points() const;
  float mean_model_texture_reprojection_error() const;
  float model_texture_inlier_fraction() const;
  const Eigen::Matrix<float, 6, 1> &model_texture_gradient() const;
  const Eigen::Matrix<float, 6, 6> &model_texture_hessian() const;
  int n_valid_model_texture_orb_points() const;
  float mean_model_texture_orb_reprojection_error() const;
  Eigen::Matrix<float, 6, 1> model_texture_orb_gradient() const;
  Eigen::Matrix<float, 6, 6> model_texture_orb_hessian() const;
  int n_valid_model_texture_disk_points() const;
  float mean_model_texture_disk_reprojection_error() const;
  const Eigen::Matrix<float, 6, 1> &model_texture_disk_gradient() const;
  const Eigen::Matrix<float, 6, 6> &model_texture_disk_hessian() const;
  int n_valid_model_photometric_points() const;
  float mean_model_photometric_residual() const;
  const Eigen::Matrix<float, 6, 1> &model_photometric_gradient() const;
  const Eigen::Matrix<float, 6, 6> &model_photometric_hessian() const;
  int n_valid_model_contour_points() const;
  float mean_model_contour_residual() const;
  const Eigen::Matrix<float, 6, 1> &model_contour_gradient() const;
  const Eigen::Matrix<float, 6, 6> &model_contour_hessian() const;
  int n_valid_model_ecc_points() const;
  float mean_model_ecc_residual() const;
  float model_ecc_correlation() const;
  const Eigen::Matrix<float, 6, 1> &model_ecc_gradient() const;
  const Eigen::Matrix<float, 6, 6> &model_ecc_hessian() const;
  float model_hypothesis_confidence() const;
  const Eigen::Matrix<float, 6, 1> &model_hypothesis_gradient() const;
  const Eigen::Matrix<float, 6, 6> &model_hypothesis_hessian() const;
  int n_valid_video_ecc_points() const;
  float mean_video_ecc_residual() const;
  float video_ecc_correlation() const;
  const Eigen::Matrix<float, 6, 1> &video_ecc_gradient() const;
  const Eigen::Matrix<float, 6, 6> &video_ecc_hessian() const;

 private:
  // Helper method for setup
  bool LoadMetaData();
  void SetUpFeatureDetectorAndMatcher();

  // Helper methods for precalculation of internal data
  void SetImshowVariables();

  // Helper methods for precalculation of referenced data and changing data
  void PrecalculateCameraVariables();
  void PrecalculateRendererVariables();
  void PrecalculatePoseVariables();
  void PrecalculateIterationDependentVariables(int corr_iteration);

  // Helper methods for CalculateCorrespondences
  void DetectAndComputeCorrKeypoints();
  void ComputeOpticalFlowCorrespondences();
  void ComputeGeometryGuidedCorrespondences();
  void ComputeModelTextureCorrespondences();
  void ComputeDiskModelTextureCorrespondences();
  void ComputeModelPhotometricData();
  void ComputeModelContourData();
  void ComputeModelEccData();
  void ComputeModelHypothesis();
  float ModelTextureGradientCorrelation();
  void ComputeVideoEccKeyframe();
  void ComputeVideoEccCorrespondences();
  bool CalculateScaleAndRegionOfInterest(cv::Rect *region_of_interest,
                                         float *scale) const;
  void ComputeKeyframeData();
  bool Reconstruct3DPoint(const cv::Point2f &center,
                          Eigen::Vector3f *center_f_body) const;
  bool IsPointValid(const Eigen::Vector3f &center_f_body,
                    bool measure_occlusions, bool model_occlusions);
  bool IsPointUnoccludedMeasured(const Eigen::Vector3f &center_f_body) const;
  bool IsPointUnoccludedModeled(const Eigen::Vector3f &center_f_body) const;

  // Helper methods for visualization
  void ShowAndSaveImage(const std::string &title, int save_index,
                        const cv::Mat &image) const;
  void VisualizePointsColorImage(const std::string &title,
                                 int save_index) const;
  void VisualizePointsDepthImage(const std::string &title,
                                 int save_index) const;
  void VisualizePointsDepthRendering(const std::string &title,
                                     int save_index) const;
  void VisualizePointsSilhouetteRendering(const std::string &title,
                                          int save_index) const;
  void VisualizeCorrespondences(const std::string &title, int save_index) const;
  void DrawPoints(const cv::Vec3b &color_point, cv::Mat *image) const;
  void DrawDepthPoints(const cv::Vec3b &color_point, cv::Mat *image) const;
  void DrawFocusedPoints(const std::shared_ptr<FocusedRenderer> &renderer_ptr,
                         const cv::Vec3b &color_point, cv::Mat *image) const;
  void DrawCorrespondences(const cv::Vec3b &color_point,
                           const cv::Vec3b &color_correspondence_point,
                           const cv::Vec3b &color_correspondence,
                           cv::Mat *image) const;

  // Helper methods for gradient calculation
  float TukeyNorm(float error);

  // Other helper methods
  bool IsSetup() const;

  // Internal data objects
  std::vector<DataPoint> data_points_;
  std::vector<PhotometricDataPoint> model_photometric_data_points_;
  std::vector<ModelContourDataPoint> model_contour_data_points_;
  std::vector<ModelEccDataPoint> model_ecc_data_points_;
  std::vector<VideoEccDataPoint> video_ecc_data_points_;
  cv::Mat video_ecc_keyframe_image_;
  cv::Mat video_ecc_keyframe_mask_;
  std::vector<Eigen::Vector3f> video_ecc_keyframe_body_points_;
  std::vector<cv::Point2f> video_ecc_keyframe_image_points_;
  int n_valid_gradient_points_ = 0;
  float mean_reprojection_error_ = 0.0f;
  float inlier_fraction_ = 0.0f;
  size_t n_model_texture_correspondence_points_ = 0;
  int n_valid_model_texture_points_ = 0;
  float mean_model_texture_reprojection_error_ = 0.0f;
  float model_texture_inlier_fraction_ = 0.0f;
  Eigen::Matrix<float, 6, 1> model_texture_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> model_texture_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  int n_valid_model_texture_disk_points_ = 0;
  float mean_model_texture_disk_reprojection_error_ = 0.0f;
  Eigen::Matrix<float, 6, 1> model_texture_disk_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> model_texture_disk_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  int n_valid_model_photometric_points_ = 0;
  float mean_model_photometric_residual_ = 0.0f;
  Eigen::Matrix<float, 6, 1> model_photometric_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> model_photometric_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  int n_valid_model_contour_points_ = 0;
  float mean_model_contour_residual_ = 0.0f;
  Eigen::Matrix<float, 6, 1> model_contour_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> model_contour_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  int n_valid_model_ecc_points_ = 0;
  float mean_model_ecc_residual_ = 0.0f;
  float model_ecc_correlation_ = 0.0f;
  Eigen::Matrix<float, 6, 1> model_ecc_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> model_ecc_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  float model_hypothesis_confidence_ = 0.0f;
  Transform3fA model_hypothesis_target_pose_{Transform3fA::Identity()};
  Eigen::Matrix<float, 6, 1> model_hypothesis_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> model_hypothesis_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  int n_valid_video_ecc_points_ = 0;
  float mean_video_ecc_residual_ = 0.0f;
  float video_ecc_correlation_ = 0.0f;
  Eigen::Matrix<float, 6, 1> video_ecc_gradient_ =
      Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6> video_ecc_hessian_ =
      Eigen::Matrix<float, 6, 6>::Zero();
  cv::Mat model_photometric_gray_image_;
  cv::Mat model_photometric_gradient_x_;
  cv::Mat model_photometric_gradient_y_;
  float model_photometric_scale_ = 1.0f;
  float model_photometric_offset_ = 0.0f;
  cv::Ptr<cv::Feature2D> feature_detector_;
  cv::Ptr<cv::Feature2D> feature_descriptor_;
  cv::Ptr<cv::Feature2D> model_texture_feature_detector_;
  cv::Ptr<cv::Feature2D> model_texture_sift_detector_;
  cv::Ptr<cv::Feature2D> model_texture_akaze_detector_;
#ifdef USE_ONNXRUNTIME
  std::unique_ptr<DiskFeatureExtractor> disk_feature_extractor_;
#endif
  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
  std::deque<std::vector<Eigen::Vector3f>> points_keyframes_;
  std::deque<cv::Mat> descriptors_keyframes_;
  std::deque<std::vector<cv::Point2f>> image_points_keyframes_;
  std::deque<cv::Mat> gray_images_keyframes_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
  Eigen::Vector3f orientation_last_keyframe_{};
  int keyframe_age_ = 0;

#ifdef USE_CUDA
  // Internal data specific to cuda implementation
  cv::Ptr<cv::cuda::ORB> feature_detector_orb_cuda_;
  cv::Ptr<cv::cuda::DescriptorMatcher> descriptor_matcher_cuda_;
  std::deque<cv::cuda::GpuMat> descriptors_keyframes_cuda_;
  cv::cuda::GpuMat descriptors_cuda_;
#endif

  // Pointers to referenced objects
  std::shared_ptr<ColorCamera> color_camera_ptr_ = nullptr;
  std::shared_ptr<DepthCamera> depth_camera_ptr_ = nullptr;
  std::shared_ptr<FocusedSilhouetteRenderer> silhouette_renderer_ptr_ = nullptr;
  std::shared_ptr<FocusedDepthRenderer> depth_renderer_ptr_ = nullptr;
  std::shared_ptr<FocusedTexturedRenderer> textured_renderer_ptr_ = nullptr;

  // Parameters for general settings
  DescriptorType descriptor_type_ = DescriptorType::ORB;
  int focused_image_size_ = 200;
  bool use_optical_flow_tracking_ = false;
  bool use_geometry_guided_matching_ = false;
  bool use_model_texture_matching_ = false;
  bool use_model_texture_sift_ = false;
  bool use_model_texture_akaze_ = false;
  bool use_model_texture_disk_ = false;
  bool use_adaptive_model_texture_disk_ = false;
  bool use_model_texture_disk_fusion_ = false;
  bool model_texture_statistics_available_ = false;
  bool adaptive_model_texture_disk_selected_ = false;
  int adaptive_model_texture_disk_evidence_ = 0;
  bool use_model_photometric_tracking_ = false;
  bool use_model_contour_tracking_ = false;
  bool use_model_ecc_tracking_ = false;
  bool use_model_hypothesis_search_ = false;
  bool use_video_ecc_tracking_ = false;
  int n_model_photometric_points_ = 120;
  float model_photometric_tukey_constant_ = 30.0f;
  float model_photometric_variance_ = 100.0f;
  int n_model_contour_points_ = 120;
  float model_contour_search_radius_ = 8.0f;
  float model_contour_tukey_constant_ = 8.0f;
  float model_contour_variance_ = 4.0f;
  int n_model_ecc_points_ = 80;
  float model_ecc_tukey_constant_ = 8.0f;
  float model_ecc_variance_ = 4.0f;
  int n_video_ecc_points_ = 80;
  float video_ecc_tukey_constant_ = 10.0f;
  float video_ecc_variance_ = 4.0f;
  float model_texture_search_radius_ = 12.0f;
  float model_texture_ratio_threshold_ = 0.75f;
  float model_texture_max_descriptor_distance_ = 64.0f;
  float model_texture_disk_similarity_threshold_ = 0.70f;
  float model_texture_disk_ransac_threshold_ = 3.0f;
  int model_texture_disk_input_size_ = 128;
  bool use_adaptive_model_texture_geometry_ = false;
  bool use_model_texture_pnp_geometry_ = false;
  std::filesystem::path model_texture_disk_model_path_{};
  std::filesystem::path model_texture_debug_directory_{};
  bool model_texture_debug_images_saved_ = false;
  float descriptor_distance_threshold_ = 0.7f;
  float tukey_norm_constant_ = 20.0f;
  std::vector<float> standard_deviations_{15.0f, 5.0f};
  float max_keyframe_rotation_difference_ = 10.0f * kPi / 180.0f;
  int max_keyframe_age_ = 100;
  int n_keyframes_ = 1;

  // Parameters for feature detection
  int orb_n_features_ = 300;
  float orb_scale_factor_ = 1.2f;
  int orb_n_levels_ = 3;
  int brisk_threshold_ = 25;
  int brisk_octave_ = 3;
  float brisk_pattern_scale_ = 0.6f;
  float daisy_radius_ = 7.0f;
  int daisy_q_radius_ = 3;
  int daisy_q_theta_ = 4;
  int daisy_q_hist_ = 8;
  bool freak_orientation_normalized_ = true;
  bool freak_scale_normalized_ = true;
  float freak_pattern_scale_ = 18.0f;
  int freak_n_octaves_ = 4;
  int sift_n_features_ = 0;
  int sift_n_octave_layers_ = 5;
  double sift_contrast_threshold_ = 0.04;
  double sift_edge_threshold_ = 10.0;
  double sift_sigma_ = 0.7;

  // Parameters for occlusion handling
  bool measure_occlusions_ = false;
  float measured_occlusion_radius_ = 0.01f;
  float measured_occlusion_threshold_ = 0.03f;
  bool model_occlusions_ = false;
  float modeled_occlusion_radius_ = 0.01f;
  float modeled_occlusion_threshold_ = 0.03f;

  // Parameters to turn on individual visualizations
  bool visualize_correspondences_correspondence_ = false;
  bool visualize_correspondences_optimization_ = false;
  bool visualize_points_result_ = false;
  bool visualize_points_depth_image_result_ = false;
  bool visualize_points_silhouette_rendering_result_ = false;
  bool visualize_points_depth_rendering_result_ = false;
  bool collect_diagnostics_ = false;

  // Parameters for visualizations
  float visualization_min_depth_ = 0.0f;
  float visualization_max_depth_ = 1.0f;

  // Precalculated variables for camera (referenced data)
  float fu_{};
  float fv_{};
  float ppu_{};
  float ppv_{};
  int image_width_minus_1_{};
  int image_height_minus_1_{};
  float depth_fu_{};
  float depth_fv_{};
  float depth_ppu_{};
  float depth_ppv_{};
  float depth_scale_{};
  int depth_image_width_minus_1_{};
  int depth_image_height_minus_1_{};

  // Precalculated variables for renderer (referenced data)
  int silhouette_image_size_minus_1_{};
  int depth_image_size_minus_1_{};

  // Precalculated variables for poses (continuously changing)
  Transform3fA body2camera_pose_{};
  Transform3fA camera2body_pose_{};
  Eigen::Matrix3f body2camera_rotation_{};
  Transform3fA body2depth_camera_pose_{};

  // Precalculate variables depending on iteration (continuously changing)
  float variance_{};
};

}  // namespace m3t

#endif  // M3T_INCLUDE_M3T_TEXTURE_MODALITY_H_
