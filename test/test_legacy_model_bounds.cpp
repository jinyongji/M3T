// Pure geometry regression; requires no OpenGL, OpenCV or dataset images.
// The Python test compiles this with the exact function from baseline.cpp.
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
void Require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
// LEGACY_BOUNDS_IMPLEMENTATION
int main() {
  constexpr double R=.8, N=2000, safety=20;
  // Includes smallest and largest measured radii among all 15 industrial CADs.
  for(double radius:{.001,.058309497936352594,.24701150201537547,.3354178407651501,.56}) {
    const double enclosing=static_cast<float>(2*radius);
    radius=.5*enclosing;
    const double diameter=LegacyModelDiameter(static_cast<float>(enclosing));
    const double focal=(N-safety)*R/diameter;
    const double tangent_radius_px=focal*radius/std::sqrt(R*R-radius*radius);
    Require(tangent_radius_px<N/2-20,"Perspective silhouette can touch image border");
    Require(R-diameter/2>0,"Invalid near clip");
    Require(diameter/2>=radius,"Depth clip excludes geometry");
    // Sample the entire meridian, including forward and rear hemispheres.
    for(int i=0;i<=10000;++i) {
      const double angle=2*3.14159265358979323846*i/10000;
      const double x=radius*std::sin(angle), z=R+radius*std::cos(angle);
      const double u=N/2+focal*x/z;
      Require(u>1 && u<N-2,"Projected sphere touches silhouette boundary");
    }
  }
  const double r=.24701150201537547;
  Require((N-safety)*R/(2*r)*r/std::sqrt(R*R-r*r)>N/2,
          "Regression fixture should reproduce the original framing failure");
  for(float invalid:{0.0f,-1.0f,1.4f,1.6f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    bool rejected=false;
    try {LegacyModelDiameter(invalid);} catch(const std::runtime_error &) {rejected=true;}
    Require(rejected,"Invalid/near-clipping bound was accepted");
  }
  std::cout<<"Perspective bound regression passed"<<std::endl;
}
