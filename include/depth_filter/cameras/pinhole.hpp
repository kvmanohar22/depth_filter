#ifndef _DEPTH_FILTER_PINHOLE_CAMERA_HPP_
#define _DEPTH_FILTER_PINHOLE_CAMERA_HPP_

#include "depth_filter/cameras/abstract.hpp"

namespace df {

class Pinhole : public AbstractCamera {
private:
  const double fx_, fy_;
  const double cx_, cy_;
  bool distortion_;
  double d_[5];
  Matrix3d K_;
  cv::Mat cvK_, cvD_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pinhole(double width, double height,
          double fx, double fy, double cx, double cy,
          double d0=0.0f, double d1=0.0f, double d2=0.0f, 
          double d3=0.0f, double d4=0.0f);
  ~Pinhole() override;

  Vector3d cam2world(const double &u, const double &v) const override;
  Vector3d cam2world(const Vector2d &px) const override;
  Vector2d world2cam(const Vector3d &xyz) const override;
  Vector2d world2cam(const Vector2d &uv) const override;

  inline double fx() const { return fx_; }
  inline double fy() const { return fy_; }
  inline double cx() const { return cx_; }
  inline double cy() const { return cy_; }

  inline double d0() { return d_[0]; }
  inline double d1() { return d_[1]; }
  inline double d2() { return d_[2]; }
  inline double d3() { return d_[3]; }
  inline double d4() { return d_[4]; }

  inline double error2() const override { return fabs(fx_); }

  inline bool distortion() { return distortion_; }

  inline cv::Mat K() { return cvK_.clone(); }
  inline cv::Mat D() { return cvD_.clone(); }
};

} // namespace df

#endif
