#include "rgb_geometry.h"
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

void Check(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
int main() {
  try {
    cv::setNumThreads(1);cv::setRNGSeed(0);
    const cv::Matx33d camera(650,0,320,0,647,260,0,0,1);
    std::mt19937 rng(20261024);std::uniform_real_distribution<float> u(-.04f,.04f);
    std::vector<cv::Point3f> object;for(int i=0;i<100;++i)object.emplace_back(u(rng),u(rng),u(rng));
    std::vector<cv::Point2f> image;
    cv::projectPoints(object,cv::Vec3d(.07,-.04,.02),cv::Vec3d(.01,-.005,.6),camera,cv::noArray(),image);
    Check(tless_rgb::VerifyVideo(object,image,camera).inliers.size()==100,"Exact RGB correspondences rejected");
    auto corrupted=image;
    std::uniform_real_distribution<float> x(0,640),y(0,520);
    for(int i=75;i<100;++i)corrupted[i]={x(rng),y(rng)};
    auto result=tless_rgb::VerifyVideo(object,corrupted,camera);
    Check(result.accepted&&result.inliers.size()>=74,"Recoverable outliers rejected");
    for(int i:result.inliers)Check(i<75,"Outlier retained");
    Check(!tless_rgb::VerifyVideo(std::vector<cv::Point3f>(object.begin(),object.begin()+5),std::vector<cv::Point2f>(image.begin(),image.begin()+5),camera).accepted,"Insufficient support accepted");
    Check(!tless_rgb::VerifyVideo(object,{},camera).accepted,"Mismatched inputs accepted");
    auto invalid=object;invalid[0].x=std::numeric_limits<float>::quiet_NaN();
    Check(!tless_rgb::VerifyVideo(invalid,image,camera).accepted,"Nonfinite input accepted");
    std::vector<cv::Point3f> degenerate(100,cv::Point3f(0,0,0));
    Check(!tless_rgb::VerifyVideo(degenerate,image,camera).accepted,"Degenerate model points accepted");
    for(int i=25;i<100;++i)corrupted[i]={x(rng),y(rng)};
    Check(!tless_rgb::VerifyVideo(object,corrupted,camera).accepted,"Insufficient inlier ratio accepted");
    std::cout<<"PASS: 7 native RGB geometric verification checks\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
