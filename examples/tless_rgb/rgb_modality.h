#pragma once

#include "rgb_geometry.h"
#include <m3t/texture_modality.h>

namespace tless_rgb {
// The generated adapter header adds one inline accessor, without changing
// TextureModality layout or any original implementation file.
class VerifiedTexture : public m3t::TextureModality {
 public:
  using m3t::TextureModality::TextureModality;
  bool verify_video = false;
  bool region_control = false;
  int candidates = 0, retained = 0;
  bool verified = false;

  bool CalculateCorrespondences(int iteration, int corr_iteration) override {
    if (!m3t::TextureModality::CalculateCorrespondences(iteration, corr_iteration)) return false;
    if (corr_iteration != 0) return true;
    auto& points = RgbAdapterPoints();
    candidates = int(std::count_if(points.begin(), points.end(),
        [](const auto& p) { return !p.from_model_texture; }));
    retained = candidates; verified = false;
    if (!verify_video && !region_control) return true;
    Verification check;
    if (verify_video && !region_control) {
      std::vector<cv::Point3f> object;
      std::vector<cv::Point2f> image;
      for (const auto& p : points) if (!p.from_model_texture) {
        object.emplace_back(p.center_f_body.x(), p.center_f_body.y(), p.center_f_body.z());
        image.emplace_back(p.correspondence_center.x(), p.correspondence_center.y());
      }
      const auto& k = color_camera_ptr()->intrinsics();
      check = VerifyVideo(object, image, cv::Matx33d(k.fu,0,k.ppu,0,k.fv,k.ppv,0,0,1));
    }
    int index = 0;
    points.erase(std::remove_if(points.begin(), points.end(), [&](const auto& p) {
      if (p.from_model_texture) return false;
      return !std::binary_search(check.inliers.begin(), check.inliers.end(), index++);
    }), points.end());
    retained = int(check.inliers.size()); verified = check.accepted;
    return true;
  }
};
}  // namespace tless_rgb
