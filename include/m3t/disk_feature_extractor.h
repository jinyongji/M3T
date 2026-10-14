// SPDX-License-Identifier: MIT

#ifndef M3T_INCLUDE_M3T_DISK_FEATURE_EXTRACTOR_H_
#define M3T_INCLUDE_M3T_DISK_FEATURE_EXTRACTOR_H_

#ifdef USE_ONNXRUNTIME

#include <filesystem/filesystem.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <memory>
#include <vector>

namespace m3t {

// Thin ONNX Runtime wrapper for the standalone DISK local-feature network.
// The network returns L2-normalized descriptors and keypoints in the resized
// input coordinate system.
class DiskFeatureExtractor {
 public:
  bool SetUp(const std::filesystem::path &model_path, int input_size,
             bool use_cuda);
  bool DetectAndCompute(const cv::Mat &image,
                        std::vector<cv::KeyPoint> *keypoints,
                        cv::Mat *descriptors);

  int input_size() const { return input_size_; }
  bool set_up() const { return set_up_; }

 private:
  bool PreloadCudaDependencies(
      const std::filesystem::path &environment_directory) const;

  bool set_up_ = false;
  int input_size_ = 128;
  bool normalized_float_keypoints_ = false;
  std::unique_ptr<Ort::Env> environment_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<float> input_buffer_;
};

}  // namespace m3t

#endif  // USE_ONNXRUNTIME
#endif  // M3T_INCLUDE_M3T_DISK_FEATURE_EXTRACTOR_H_
