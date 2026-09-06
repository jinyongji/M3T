// SPDX-License-Identifier: MIT
#include <m3t/loader_camera.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
double Rmse(const cv::Mat &image, const cv::Mat &reference) {
  return cv::norm(image, reference, cv::NORM_L2) /
         std::sqrt(double(image.total() * image.channels()));
}
}  // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 2, "Usage: test_offline_conditioning <fresh-output-directory>");
    const std::filesystem::path directory{argv[1]};
    std::filesystem::create_directories(directory);
    const auto first = directory / "fixture_0000.png";
    const auto second = directory / "fixture_0001.png";
    Require(!std::filesystem::exists(first) && !std::filesystem::exists(second),
            "Use a fresh directory; existing fixtures will not be overwritten");
    const cv::Mat clean(128, 160, CV_8UC3, cv::Scalar::all(128));
    cv::Mat noise(clean.size(), CV_32FC3), noisy_float, noisy;
    cv::RNG random(314159);
    random.fill(noise, cv::RNG::NORMAL, 0.0, 10.0);
    clean.convertTo(noisy_float, CV_32F);
    noisy_float += noise;
    noisy_float.convertTo(noisy, CV_8U);
    Require(cv::imwrite(first.string(), noisy), "Cannot save noisy fixture");
    Require(cv::imwrite(second.string(), clean), "Cannot save clean fixture");
    const m3t::Intrinsics intrinsics{100, 100, 80, 64, 160, 128};
    for (int mode = 0; mode < 3; ++mode) {
      m3t::LoaderColorCamera camera{"fixture", directory, intrinsics,
                                    "fixture_", 0, 4};
      camera.set_use_noise_adaptive_filtering(mode != 0);
      camera.set_use_gaussian_noise_filter(mode == 1);
      Require(camera.SetUp(), "Cannot load noisy fixture");
      Require(cv::norm(camera.unfiltered_image(), noisy, cv::NORM_INF) == 0,
              "Filtering changed the raw texture observation");
      if (mode == 0) {
        Require(cv::norm(camera.image(), noisy, cv::NORM_INF) == 0,
                "Disabled mode changed legacy input");
        Require(camera.preprocessing_time_us() == 0, "Disabled timer is not zero");
      } else {
        Require(camera.noise_standard_deviation() > 7 &&
                    camera.noise_standard_deviation() < 13,
                "Noise estimate is outside synthetic tolerance");
        Require(Rmse(camera.image(), clean) < 0.6 * Rmse(noisy, clean),
                "Conditioning did not reduce synthetic flat-field noise");
        Require(camera.preprocessing_time_us() > 0, "Conditioning time not recorded");
      }
      Require(camera.UpdateImage(true), "Cannot load clean fixture");
      Require(cv::norm(camera.image(), clean, cv::NORM_INF) == 0 &&
                  cv::norm(camera.unfiltered_image(), clean, cv::NORM_INF) == 0,
              "Clean bypass or frame-buffer lifetime is incorrect");
      Require(camera.noise_standard_deviation() == 0, "Clean noise state not reset");
      std::cout << "PASS mode=" << mode
                << " raw preservation, conditioning, clean bypass, next frame\n";
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
