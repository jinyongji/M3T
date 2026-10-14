// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include <m3t/loader_camera.h>

#include <opencv2/core/eigen.hpp>

#include <array>
#include <chrono>

namespace m3t {

LoaderColorCamera::LoaderColorCamera(
    const std::string &name, const std::filesystem::path &load_directory,
    const Intrinsics &intrinsics, const std::string &image_name_pre,
    int load_index, int n_leading_zeros, const std::string &image_name_post,
    const std::string &load_image_type)
    : ColorCamera{name},
      load_directory_{load_directory},
      image_name_pre_{image_name_pre},
      load_index_{load_index},
      n_leading_zeros_{n_leading_zeros},
      image_name_post_{image_name_post},
      load_image_type_{load_image_type} {
  intrinsics_ = intrinsics;
}

LoaderColorCamera::LoaderColorCamera(const std::string &name,
                                     const std::filesystem::path &metafile_path)
    : ColorCamera{name, metafile_path} {}

bool LoaderColorCamera::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;
  SaveMetaDataIfDesired();
  set_up_ = true;
  return UpdateImage(true);
}

void LoaderColorCamera::set_load_directory(
    const std::filesystem::path &load_directory) {
  load_directory_ = load_directory;
  set_up_ = false;
}

void LoaderColorCamera::set_intrinsics(const Intrinsics &intrinsics) {
  intrinsics_ = intrinsics;
  set_up_ = false;
}

void LoaderColorCamera::set_image_name_pre(const std::string &image_name_pre) {
  image_name_pre_ = image_name_pre;
  set_up_ = false;
}

void LoaderColorCamera::set_load_index(int load_index) {
  load_index_ = load_index;
  set_up_ = false;
}

void LoaderColorCamera::set_n_leading_zeros(int n_leading_zeros) {
  n_leading_zeros_ = n_leading_zeros;
  set_up_ = false;
}

void LoaderColorCamera::set_image_name_post(
    const std::string &image_name_post) {
  image_name_post_ = image_name_post;
  set_up_ = false;
}

void LoaderColorCamera::set_load_image_type(
    const std::string &load_image_type) {
  load_image_type_ = load_image_type;
  set_up_ = false;
}

bool LoaderColorCamera::UpdateImage(bool synchronized) {
  if (!set_up_) {
    std::cerr << "Set up loader color camera " << name_ << " first"
              << std::endl;
    return false;
  }

  int n_zeros =
      std::max(n_leading_zeros_ - int(std::to_string(load_index_).length()), 0);
  std::filesystem::path path{load_directory_ /
                             (image_name_pre_ + std::string(n_zeros, '0') +
                              std::to_string(load_index_) + image_name_post_ +
                              "." + load_image_type_)};
  image_ = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  if (image_.empty()) {
    std::cerr << "Could not read image from " << path.string() << std::endl;
    return false;
  }

  unfiltered_image_ = image_;
  FilterImageIfRequested();
  load_index_++;
  SaveImageIfDesired();
  return true;
}

void LoaderColorCamera::FilterImageIfRequested() {
  preprocessing_time_us_ = 0.0f;
  noise_standard_deviation_ = 0.0f;
  if (!use_noise_adaptive_filtering_ || image_.type() != CV_8UC3 ||
      image_.rows < 2 || image_.cols < 2)
    return;
  const auto begin = std::chrono::steady_clock::now();

  // Robust Haar high-pass MAD: (a-b-c+d)/2 has unit noise gain. A sparse
  // histogram avoids sorting and rejects isolated texture/contour responses.
  std::array<int, 511> histogram{};
  int samples = 0;
  for (int v = 0; v + 1 < image_.rows; v += 4) {
    const auto *row0 = image_.ptr<cv::Vec3b>(v);
    const auto *row1 = image_.ptr<cv::Vec3b>(v + 1);
    for (int u = 0; u + 1 < image_.cols; u += 4) {
      for (int c = 0; c < 3; ++c) {
        const int response = int(row0[u][c]) - int(row0[u + 1][c]) -
                             int(row1[u][c]) + int(row1[u + 1][c]);
        ++histogram[size_t(std::abs(response))];
        ++samples;
      }
    }
  }
  int cumulative = 0;
  for (size_t i = 0; i < histogram.size(); ++i) {
    cumulative += histogram[i];
    if (cumulative >= (samples + 1) / 2) {
      noise_standard_deviation_ = 0.7413f * float(i);
      break;
    }
  }

  // Local Wiener shrinkage preserves high-variance structures and denoises
  // flat surfaces. Clean images bypass it, keeping their original evidence.
  if (noise_standard_deviation_ > 4.0f && use_gaussian_noise_filter_) {
    cv::Mat filtered;
    cv::GaussianBlur(image_, filtered, cv::Size{5, 5}, 1.0);
    image_ = std::move(filtered);
  } else if (noise_standard_deviation_ > 4.0f) {
    cv::Mat image_float, mean, second_moment, variance, gain, restored;
    image_.convertTo(image_float, CV_32F);
    cv::boxFilter(image_float, mean, -1, cv::Size{5, 5});
    cv::sqrBoxFilter(image_float, second_moment, CV_32F, cv::Size{5, 5});
    variance = second_moment - mean.mul(mean);
    const float noise_variance =
        noise_standard_deviation_ * noise_standard_deviation_;
    cv::max(variance, noise_variance, variance);
    cv::divide(noise_variance, variance, gain);
    gain = cv::Scalar::all(1.0) - gain;
    restored = mean + gain.mul(image_float - mean);
    cv::Mat filtered;
    restored.convertTo(filtered, CV_8U);
    image_ = std::move(filtered);
  }
  preprocessing_time_us_ = float(std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - begin).count());
}

const std::filesystem::path &LoaderColorCamera::load_directory() const {
  return load_directory_;
}

const std::string &LoaderColorCamera::image_name_pre() const {
  return image_name_pre_;
}

int LoaderColorCamera::load_index() const { return load_index_; }

int LoaderColorCamera::n_leading_zeros() const { return n_leading_zeros_; }

const std::string &LoaderColorCamera::image_name_post() const {
  return image_name_post_;
}

const std::string &LoaderColorCamera::load_image_type() const {
  return load_image_type_;
}

bool LoaderColorCamera::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  if (!(ReadRequiredValueFromYaml(fs, "load_directory", &load_directory_) &&
        ReadRequiredValueFromYaml(fs, "intrinsics", &intrinsics_))) {
    std::cerr << "Could not read all required body parameters from "
              << metafile_path_ << std::endl;
    return false;
  }
  ReadOptionalValueFromYaml(fs, "camera2world_pose", &camera2world_pose_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_index", &save_index_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
  ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
  ReadOptionalValueFromYaml(fs, "image_name_pre", &image_name_pre_);
  ReadOptionalValueFromYaml(fs, "load_index", &load_index_);
  ReadOptionalValueFromYaml(fs, "n_leading_zeros", &n_leading_zeros_);
  ReadOptionalValueFromYaml(fs, "image_name_post", &image_name_post_);
  ReadOptionalValueFromYaml(fs, "load_image_type", &load_image_type_);
  fs.release();

  // Process parameters
  if (load_directory_.is_relative())
    load_directory_ = metafile_path_.parent_path() / load_directory_;
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  world2camera_pose_ = camera2world_pose_.inverse();
  return true;
}

LoaderDepthCamera::LoaderDepthCamera(
    const std::string &name, const std::filesystem::path &load_directory,
    const Intrinsics &intrinsics, float depth_scale,
    const std::string &image_name_pre, int load_index, int n_leading_zeros,
    const std::string &image_name_post, const std::string &load_image_type)
    : DepthCamera{name},
      load_directory_{load_directory},
      image_name_pre_{image_name_pre},
      load_index_{load_index},
      n_leading_zeros_{n_leading_zeros},
      image_name_post_{image_name_post},
      load_image_type_{load_image_type} {
  intrinsics_ = intrinsics;
  depth_scale_ = depth_scale;
}

LoaderDepthCamera::LoaderDepthCamera(const std::string &name,
                                     const std::filesystem::path &metafile_path)
    : DepthCamera{name, metafile_path} {}

bool LoaderDepthCamera::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;
  SaveMetaDataIfDesired();
  set_up_ = true;
  return UpdateImage(true);
}

void LoaderDepthCamera::set_load_directory(
    const std::filesystem::path &load_directory) {
  load_directory_ = load_directory;
  set_up_ = false;
}

void LoaderDepthCamera::set_intrinsics(const Intrinsics &intrinsics) {
  intrinsics_ = intrinsics;
  set_up_ = false;
}

void LoaderDepthCamera::set_depth_scale(float depth_scale) {
  depth_scale_ = depth_scale;
  set_up_ = false;
}

void LoaderDepthCamera::set_image_name_pre(const std::string &image_name_pre) {
  image_name_pre_ = image_name_pre;
  set_up_ = false;
}

void LoaderDepthCamera::set_load_index(int load_index) {
  load_index_ = load_index;
  set_up_ = false;
}

void LoaderDepthCamera::set_n_leading_zeros(int n_leading_zeros) {
  n_leading_zeros_ = n_leading_zeros;
  set_up_ = false;
}

void LoaderDepthCamera::set_image_name_post(
    const std::string &image_name_post) {
  image_name_post_ = image_name_post;
  set_up_ = false;
}

void LoaderDepthCamera::set_load_image_type(
    const std::string &load_image_type) {
  load_image_type_ = load_image_type;
  set_up_ = false;
}

bool LoaderDepthCamera::UpdateImage(bool synchronized) {
  if (!set_up_) {
    std::cerr << "Set up loader depth camera " << name_ << " first"
              << std::endl;
    return false;
  }

  int n_zeros =
      std::max(n_leading_zeros_ - int(std::to_string(load_index_).length()), 0);
  std::filesystem::path path{load_directory_ /
                             (image_name_pre_ + std::string(n_zeros, '0') +
                              std::to_string(load_index_) + image_name_post_ +
                              "." + load_image_type_)};
  image_ = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  if (image_.empty()) {
    std::cerr << "Could not read image from " << path.string() << std::endl;
    return false;
  }

  load_index_++;
  SaveImageIfDesired();
  return true;
}

const std::filesystem::path &LoaderDepthCamera::load_directory() const {
  return load_directory_;
}

const std::string &LoaderDepthCamera::image_name_pre() const {
  return image_name_pre_;
}

int LoaderDepthCamera::load_index() const { return load_index_; }

int LoaderDepthCamera::n_leading_zeros() const { return n_leading_zeros_; }

const std::string &LoaderDepthCamera::image_name_post() const {
  return image_name_post_;
}

const std::string &LoaderDepthCamera::load_image_type() const {
  return load_image_type_;
}

bool LoaderDepthCamera::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  if (!(ReadRequiredValueFromYaml(fs, "load_directory", &load_directory_) &&
        ReadRequiredValueFromYaml(fs, "intrinsics", &intrinsics_) &&
        ReadRequiredValueFromYaml(fs, "depth_scale", &depth_scale_))) {
    std::cerr << "Could not read all required body parameters from "
              << metafile_path_ << std::endl;
    return false;
  }
  ReadOptionalValueFromYaml(fs, "camera2world_pose", &camera2world_pose_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_index", &save_index_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
  ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
  ReadOptionalValueFromYaml(fs, "image_name_pre", &image_name_pre_);
  ReadOptionalValueFromYaml(fs, "load_index", &load_index_);
  ReadOptionalValueFromYaml(fs, "n_leading_zeros", &n_leading_zeros_);
  ReadOptionalValueFromYaml(fs, "image_name_post", &image_name_post_);
  ReadOptionalValueFromYaml(fs, "load_image_type", &load_image_type_);
  fs.release();

  // Process parameters
  if (load_directory_.is_relative())
    load_directory_ = metafile_path_.parent_path() / load_directory_;
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  world2camera_pose_ = camera2world_pose_.inverse();
  return true;
}

}  // namespace m3t
