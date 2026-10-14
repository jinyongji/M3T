// SPDX-License-Identifier: MIT

#include <m3t/disk_feature_extractor.h>

#ifdef USE_ONNXRUNTIME

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace m3t {

bool DiskFeatureExtractor::SetUp(const std::filesystem::path &model_path,
                                 int input_size, bool use_cuda) {
  set_up_ = false;
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "DISK model does not exist: " << model_path << std::endl;
    return false;
  }
  if (input_size <= 0) {
    std::cerr << "DISK input size must be positive" << std::endl;
    return false;
  }

  try {
    if (use_cuda && !PreloadCudaDependencies(model_path.parent_path())) {
      return false;
    }
    environment_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                               "m3t_disk");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    if (use_cuda) {
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
    }
#ifdef _WIN32
    session_ = std::make_unique<Ort::Session>(*environment_,
                                              model_path.c_str(), options);
#else
    const std::string model_path_string{model_path.string()};
    session_ = std::make_unique<Ort::Session>(
        *environment_, model_path_string.c_str(), options);
#endif
    input_size_ = input_size;
    const auto keypoint_type{
        session_->GetOutputTypeInfo(0)
            .GetTensorTypeAndShapeInfo()
            .GetElementType()};
    normalized_float_keypoints_ =
        keypoint_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    input_buffer_.resize(size_t(3 * input_size_ * input_size_));
  } catch (const Ort::Exception &exception) {
    std::cerr << "Could not initialize DISK ONNX model: " << exception.what()
              << std::endl;
    session_.reset();
    environment_.reset();
    return false;
  }

  set_up_ = true;
  return true;
}

bool DiskFeatureExtractor::DetectAndCompute(
    const cv::Mat &image, std::vector<cv::KeyPoint> *keypoints,
    cv::Mat *descriptors) {
  keypoints->clear();
  descriptors->release();
  if (!set_up_ || image.empty()) return false;

  cv::Mat rgb_image;
  if (image.channels() == 4) {
    cv::cvtColor(image, rgb_image, cv::COLOR_BGRA2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 1) {
    cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
  } else {
    std::cerr << "DISK expects a one-, three-, or four-channel image"
              << std::endl;
    return false;
  }
  cv::resize(rgb_image, rgb_image, cv::Size{input_size_, input_size_}, 0.0,
             0.0, cv::INTER_LINEAR);

  const int area{input_size_ * input_size_};
  for (int row = 0; row < input_size_; ++row) {
    const auto *pixels{rgb_image.ptr<cv::Vec3b>(row)};
    for (int column = 0; column < input_size_; ++column) {
      const int index{row * input_size_ + column};
      input_buffer_[size_t(index)] = float(pixels[column][0]) / 255.0f;
      input_buffer_[size_t(area + index)] =
          float(pixels[column][1]) / 255.0f;
      input_buffer_[size_t(2 * area + index)] =
          float(pixels[column][2]) / 255.0f;
    }
  }

  try {
    const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    Ort::MemoryInfo memory_info{
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Value input_tensor{Ort::Value::CreateTensor<float>(
        memory_info, input_buffer_.data(), input_buffer_.size(),
        input_shape.data(), input_shape.size())};
    constexpr std::array<const char *, 1> input_names{"image"};
    constexpr std::array<const char *, 3> output_names{
        "keypoints", "scores", "descriptors"};
    auto outputs{session_->Run(Ort::RunOptions{nullptr}, input_names.data(),
                               &input_tensor, input_names.size(),
                               output_names.data(), output_names.size())};

    const auto keypoint_shape{
        outputs[0].GetTensorTypeAndShapeInfo().GetShape()};
    const auto descriptor_shape{
        outputs[2].GetTensorTypeAndShapeInfo().GetShape()};
    const bool batched_outputs{
        keypoint_shape.size() == 3 && descriptor_shape.size() == 3 &&
        keypoint_shape[0] == 1 && keypoint_shape[2] == 2 &&
        descriptor_shape[0] == 1 &&
        keypoint_shape[1] == descriptor_shape[1]};
    const bool unbatched_outputs{
        keypoint_shape.size() == 2 && descriptor_shape.size() == 2 &&
        keypoint_shape[1] == 2 &&
        keypoint_shape[0] == descriptor_shape[0]};
    if (!batched_outputs && !unbatched_outputs) {
      std::cerr << "Unexpected DISK output shape" << std::endl;
      return false;
    }
    const int n_keypoints{
        int(batched_outputs ? keypoint_shape[1] : keypoint_shape[0])};
    const int descriptor_dimension{
        int(batched_outputs ? descriptor_shape[2] : descriptor_shape[1])};
    const float *score_data{outputs[1].GetTensorData<float>()};
    const float *descriptor_data{outputs[2].GetTensorData<float>()};
    keypoints->reserve(size_t(n_keypoints));
    if (normalized_float_keypoints_) {
      const float *keypoint_data{outputs[0].GetTensorData<float>()};
      for (int index = 0; index < n_keypoints; ++index) {
        keypoints->emplace_back(
            (keypoint_data[2 * index] + 1.0f) * 0.5f * float(input_size_),
            (keypoint_data[2 * index + 1] + 1.0f) * 0.5f *
                float(input_size_),
            1.0f, -1.0f, score_data[index]);
      }
    } else {
      const int64_t *keypoint_data{outputs[0].GetTensorData<int64_t>()};
      for (int index = 0; index < n_keypoints; ++index) {
        keypoints->emplace_back(float(keypoint_data[2 * index]),
                                float(keypoint_data[2 * index + 1]), 1.0f,
                                -1.0f, score_data[index]);
      }
    }
    cv::Mat descriptor_view{n_keypoints, descriptor_dimension, CV_32F,
                            const_cast<float *>(descriptor_data)};
    descriptor_view.copyTo(*descriptors);
  } catch (const Ort::Exception &exception) {
    std::cerr << "DISK inference failed: " << exception.what() << std::endl;
    return false;
  }
  return true;
}

bool DiskFeatureExtractor::PreloadCudaDependencies(
    const std::filesystem::path &environment_directory) const {
#ifdef _WIN32
  const auto nvidia_directory{environment_directory / "Lib" /
                              "site-packages" / "nvidia"};
  if (!std::filesystem::exists(nvidia_directory)) {
    std::cerr << "CUDA wheel directory does not exist: " << nvidia_directory
              << std::endl;
    return false;
  }

  static std::vector<DLL_DIRECTORY_COOKIE> directory_cookies;
  static std::vector<HMODULE> library_handles;
  if (!directory_cookies.empty()) return true;
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                           LOAD_LIBRARY_SEARCH_USER_DIRS);
  constexpr std::array<const wchar_t *, 6> components{
      L"cuda_runtime", L"nvjitlink", L"cublas",
      L"cufft",       L"curand",    L"cudnn"};
  std::vector<std::filesystem::path> dll_paths;
  for (const auto *component : components) {
    const auto bin_directory{nvidia_directory / component / "bin"};
    if (!std::filesystem::exists(bin_directory)) continue;
    const auto cookie{AddDllDirectory(bin_directory.c_str())};
    if (cookie) directory_cookies.push_back(cookie);
    for (const auto &entry : std::filesystem::directory_iterator{
             bin_directory}) {
      if (entry.is_regular_file() && entry.path().extension() == ".dll")
        dll_paths.push_back(entry.path());
    }
  }
  std::stable_sort(dll_paths.begin(), dll_paths.end(),
                   [](const auto &first, const auto &second) {
                     const bool first_is_cudnn_base{
                         first.filename() == "cudnn64_9.dll"};
                     const bool second_is_cudnn_base{
                         second.filename() == "cudnn64_9.dll"};
                     return first_is_cudnn_base && !second_is_cudnn_base;
                   });
  for (const auto &dll_path : dll_paths) {
    if (const HMODULE handle{LoadLibraryExW(
            dll_path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS)})
      library_handles.push_back(handle);
  }
  return !library_handles.empty();
#else
  (void)environment_directory;
  return true;
#endif
}

}  // namespace m3t

#endif  // USE_ONNXRUNTIME
