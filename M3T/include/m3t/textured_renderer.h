// SPDX-License-Identifier: MIT
// Copyright (c) 2026

#ifndef M3T_INCLUDE_M3T_TEXTURED_RENDERER_H_
#define M3T_INCLUDE_M3T_TEXTURED_RENDERER_H_

#include <m3t/camera.h>
#include <m3t/renderer.h>
#include <m3t/renderer_geometry.h>

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>

namespace m3t {

class TexturedRendererCore {
 public:
  TexturedRendererCore() = default;
  TexturedRendererCore(const TexturedRendererCore &) = delete;
  TexturedRendererCore &operator=(const TexturedRendererCore &) = delete;
  ~TexturedRendererCore();
  bool SetUp(const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
             int image_width, int image_height);
  bool StartRendering(const Eigen::Matrix4f &projection_matrix,
                      const Transform3fA &world2camera_pose);
  bool FetchColorImage(cv::Mat *color_image);
  bool FetchDepthImage(cv::Mat *depth_image);

 private:
  void CreateBufferObjects();
  void DeleteBufferObjects();

  std::shared_ptr<RendererGeometry> renderer_geometry_ptr_;
  int image_width_ = 0;
  int image_height_ = 0;
  static std::string vertex_shader_code_;
  static std::string fragment_shader_code_;
  unsigned fbo_ = 0;
  unsigned rbo_color_ = 0;
  unsigned rbo_depth_ = 0;
  unsigned shader_program_ = 0;
  bool image_rendered_ = false;
  bool color_image_fetched_ = false;
  bool depth_image_fetched_ = false;
  bool initial_set_up_ = false;
};

class FocusedTexturedRenderer : public FocusedDepthRenderer {
 public:
  FocusedTexturedRenderer(
      const std::string &name,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
      int image_size = 200, float z_min = 0.02f, float z_max = 10.0f);
  FocusedTexturedRenderer(
      const std::string &name,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const std::shared_ptr<Camera> &camera_ptr, int image_size = 200,
      float z_min = 0.02f, float z_max = 10.0f);
  bool SetUp() override;
  bool StartRendering() override;
  bool FetchColorImage();
  bool FetchDepthImage() override;
  const cv::Mat &focused_color_image() const;

 private:
  cv::Mat focused_color_image_;
  TexturedRendererCore core_{};
};

}  // namespace m3t

#endif  // M3T_INCLUDE_M3T_TEXTURED_RENDERER_H_
