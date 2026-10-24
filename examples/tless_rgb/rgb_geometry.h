#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace tless_rgb {
struct Verification {
  std::vector<int> inliers;
  bool accepted = false;
};

// Only image/model correspondences enter this function. No reference pose,
// target visibility annotation, depth measurement or success label is used.
inline Verification VerifyVideo(const std::vector<cv::Point3f>& object,
                                const std::vector<cv::Point2f>& image,
                                const cv::Matx33d& camera) {
  Verification result;
  if (object.size() != image.size() || object.size() < 6) return result;
  for (size_t i = 0; i < object.size(); ++i) {
    const auto& p = object[i]; const auto& q = image[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(q.x) || !std::isfinite(q.y)) return result;
  }
  cv::Mat rotation, translation, inliers;
  try {
    cv::Point3d center{0,0,0};
    for (const auto& p : object) center += cv::Point3d(p);
    center *= 1.0/double(object.size());
    cv::Matx33d covariance = cv::Matx33d::zeros();
    for (const auto& p : object) {
      const cv::Vec3d d(p.x-center.x,p.y-center.y,p.z-center.z);
      covariance += d*d.t();
    }
    cv::Mat spectrum; cv::SVD::compute(covariance,spectrum);
    if (spectrum.at<double>(1) <= std::max(1e-16,spectrum.at<double>(0)*1e-6)) return result;
    if (!cv::solvePnPRansac(object, image, camera, cv::noArray(), rotation,
                           translation, false, 100, 4.0f, 0.99, inliers,
                           cv::SOLVEPNP_EPNP)) return result;
    if (!cv::checkRange(rotation) || !cv::checkRange(translation)) return result;
    cv::Mat matrix; cv::Rodrigues(rotation, matrix);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object, rotation, translation, camera, cv::noArray(), projected);
    for (int j = 0; j < int(inliers.total()); ++j) {
      const int i = inliers.at<int>(j);
      if (i < 0 || i >= int(object.size())) continue;
      const auto& p = object[i];
      const double z = matrix.at<double>(2, 0)*p.x + matrix.at<double>(2, 1)*p.y +
                       matrix.at<double>(2, 2)*p.z + translation.at<double>(2);
      const auto difference = projected[i] - image[i];
      if (z > 0.0 && difference.dot(difference) <= 16.0f)
        result.inliers.push_back(i);
    }
  } catch (const cv::Exception&) { return Verification{}; }
  std::sort(result.inliers.begin(), result.inliers.end());
  result.inliers.erase(std::unique(result.inliers.begin(), result.inliers.end()), result.inliers.end());
  result.accepted = result.inliers.size() >= 6 && 2*result.inliers.size() >= object.size();
  if (!result.accepted) result.inliers.clear();
  return result;
}
}  // namespace tless_rgb
