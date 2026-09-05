// SPDX-License-Identifier: MIT
// Copyright (c) 2026

#include <m3t/textured_renderer.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

namespace m3t {

std::string TexturedRendererCore::vertex_shader_code_ =
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 2) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "uniform mat4 Trans;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = Trans * vec4(aPos, 1.0);\n"
    "  TexCoord = aTexCoord;\n"
    "}";

std::string TexturedRendererCore::fragment_shader_code_ =
    "#version 330 core\n"
    "in vec2 TexCoord;\n"
    "out vec4 FragColor;\n"
    "uniform sampler2D TextureImage;\n"
    "void main()\n"
    "{\n"
    "  FragColor = vec4(texture(TextureImage, TexCoord).rgb, 1.0);\n"
    "}";

TexturedRendererCore::~TexturedRendererCore() {
  if (initial_set_up_) DeleteBufferObjects();
}

bool TexturedRendererCore::SetUp(
    const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
    int image_width, int image_height) {
  renderer_geometry_ptr_ = renderer_geometry_ptr;
  image_width_ = image_width;
  image_height_ = image_height;
  image_rendered_ = false;
  if (!initial_set_up_ &&
      !CreateShaderProgram(renderer_geometry_ptr_.get(),
                           vertex_shader_code_.c_str(),
                           fragment_shader_code_.c_str(), &shader_program_))
    return false;
  if (initial_set_up_) DeleteBufferObjects();
  CreateBufferObjects();
  initial_set_up_ = true;
  return true;
}

bool TexturedRendererCore::StartRendering(
    const Eigen::Matrix4f &projection_matrix,
    const Transform3fA &world2camera_pose) {
  if (!initial_set_up_) return false;
  renderer_geometry_ptr_->MakeContextCurrent();
  glViewport(0, 0, image_width_, image_height_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glFrontFace(GL_CCW);
  glCullFace(GL_FRONT);
  glUseProgram(shader_program_);
  glUniform1i(glGetUniformLocation(shader_program_, "TextureImage"), 0);
  for (const auto &render_data_body :
       renderer_geometry_ptr_->render_data_bodies()) {
    const Transform3fA trans_without_projection{
        world2camera_pose * render_data_body.body_ptr->geometry2world_pose()};
    const Eigen::Matrix4f trans{
        projection_matrix * trans_without_projection.matrix()};
    const int loc{glGetUniformLocation(shader_program_, "Trans")};
    glUniformMatrix4fv(loc, 1, GL_FALSE, trans.data());
    if (render_data_body.body_ptr->geometry_enable_culling())
      glEnable(GL_CULL_FACE);
    else
      glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, render_data_body.texture);
    glBindVertexArray(render_data_body.vao);
    glDrawArrays(GL_TRIANGLES, 0, render_data_body.n_vertices);
    glBindVertexArray(0);
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  renderer_geometry_ptr_->DetachContext();
  image_rendered_ = true;
  color_image_fetched_ = false;
  depth_image_fetched_ = false;
  return true;
}

bool TexturedRendererCore::FetchColorImage(cv::Mat *color_image) {
  if (!initial_set_up_ || !image_rendered_) return false;
  if (color_image_fetched_) return true;
  renderer_geometry_ptr_->MakeContextCurrent();
  glPixelStorei(GL_PACK_ALIGNMENT, (color_image->step & 3) ? 1 : 4);
  glPixelStorei(GL_PACK_ROW_LENGTH,
                GLint(color_image->step / color_image->elemSize()));
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glReadPixels(0, 0, image_width_, image_height_, GL_BGRA, GL_UNSIGNED_BYTE,
               color_image->data);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  renderer_geometry_ptr_->DetachContext();
  color_image_fetched_ = true;
  return true;
}

bool TexturedRendererCore::FetchDepthImage(cv::Mat *depth_image) {
  if (!initial_set_up_ || !image_rendered_) return false;
  if (depth_image_fetched_) return true;
  renderer_geometry_ptr_->MakeContextCurrent();
  glPixelStorei(GL_PACK_ALIGNMENT, (depth_image->step & 3) ? 1 : 4);
  glPixelStorei(GL_PACK_ROW_LENGTH,
                GLint(depth_image->step / depth_image->elemSize()));
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glReadPixels(0, 0, image_width_, image_height_, GL_DEPTH_COMPONENT,
               GL_UNSIGNED_SHORT, depth_image->data);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  renderer_geometry_ptr_->DetachContext();
  depth_image_fetched_ = true;
  return true;
}

void TexturedRendererCore::CreateBufferObjects() {
  renderer_geometry_ptr_->MakeContextCurrent();
  glGenRenderbuffers(1, &rbo_color_);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo_color_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, image_width_, image_height_);
  glGenRenderbuffers(1, &rbo_depth_);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, image_width_,
                        image_height_);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  glGenFramebuffers(1, &fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER, rbo_color_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, rbo_depth_);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  renderer_geometry_ptr_->DetachContext();
}

void TexturedRendererCore::DeleteBufferObjects() {
  renderer_geometry_ptr_->MakeContextCurrent();
  glDeleteRenderbuffers(1, &rbo_color_);
  glDeleteRenderbuffers(1, &rbo_depth_);
  glDeleteFramebuffers(1, &fbo_);
  renderer_geometry_ptr_->DetachContext();
}

FocusedTexturedRenderer::FocusedTexturedRenderer(
    const std::string &name,
    const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
    const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
    int image_size, float z_min, float z_max)
    : FocusedDepthRenderer{name, renderer_geometry_ptr, world2camera_pose,
                           intrinsics, image_size, z_min, z_max} {}

FocusedTexturedRenderer::FocusedTexturedRenderer(
    const std::string &name,
    const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
    const std::shared_ptr<Camera> &camera_ptr, int image_size, float z_min,
    float z_max)
    : FocusedDepthRenderer{name, renderer_geometry_ptr, camera_ptr, image_size,
                           z_min, z_max} {}

bool FocusedTexturedRenderer::SetUp() {
  const std::lock_guard<std::mutex> lock{mutex_};
  set_up_ = false;
  if (!renderer_geometry_ptr_->set_up()) {
    std::cerr << "Renderer geometry " << renderer_geometry_ptr_->name()
              << " was not set up" << std::endl;
    return false;
  }
  if (camera_ptr_ && !InitParametersFromCamera()) return false;
  if (referenced_body_ptrs_.empty()) {
    std::cerr << "No referenced bodies were assigned to renderer " << name_
              << std::endl;
    return false;
  }
  for (const auto &body_ptr : referenced_body_ptrs_) {
    if (!body_ptr->set_up()) return false;
  }
  CalculateProjectionMatrix();
  CalculateProjectionTerms();
  ClearDepthImage();
  focused_color_image_ = cv::Mat::zeros(image_size_, image_size_, CV_8UC4);
  if (!core_.SetUp(renderer_geometry_ptr_, image_size_, image_size_))
    return false;
  set_up_ = true;
  return true;
}

bool FocusedTexturedRenderer::StartRendering() {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!set_up_) return false;
  CalculateProjectionMatrix();
  return core_.StartRendering(projection_matrix_, world2camera_pose_);
}

bool FocusedTexturedRenderer::FetchColorImage() {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!set_up_) return false;
  return core_.FetchColorImage(&focused_color_image_);
}

bool FocusedTexturedRenderer::FetchDepthImage() {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!set_up_) return false;
  return core_.FetchDepthImage(&focused_depth_image_);
}

const cv::Mat &FocusedTexturedRenderer::focused_color_image() const {
  return focused_color_image_;
}

}  // namespace m3t
