#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "depth_filter/utils.h"
#include "depth_filter/cameras/pinhole.h"

namespace df {

Pinhole::Pinhole(double width, double height,
                 double fx, double fy,
                 double cx, double cy,
                 double d0, double d1, double d2,
                 double d3, double d4) :
  AbstractCamera(width, height),
  fx_(fx), fy_(fy),
  cx_(cx), cy_(cy),
  distortion_(fabs(d0) > 1e-7) {
  d_[0] = d0; d_[1] = d1; d_[2] = d2;  d_[3] = d3; d_[4] = d4;
  K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
  cvK_ = (cv::Mat_<float>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  cvD_ = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);
}

Pinhole::~Pinhole() {}

Vector3d Pinhole::cam2world(const double &u, const double &v) const {
  Vector3d xyz;
  if (!distortion_) {
    xyz[0] = (u - cx_) / fx_;
    xyz[1] = (v - cy_) / fy_;
  } else {
    cv::Point2f uv(u, v), px;
    const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
    cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
    cv::undistortPoints(src_pt, dst_pt, cvK_, cvD_);
    xyz[0] = px.x;
    xyz[1] = px.y;
  }
  xyz[2] = 1.0f;
  return xyz.normalized();
}

Vector3d Pinhole::cam2world(const Vector2d &px) const {
  return cam2world(px[0], px[1]);
}

Vector2d Pinhole::world2cam(const Vector2d &uv) const {
  Vector2d px;
  if (!distortion_) {
    px[0] = fx_ * uv[0] + cx_;
    px[1] = fy_ * uv[1] + cy_;
  } else {
    double x, y, r2, r4, r6, cdist, xd, yd, a1, a2, a3;
    x = uv[0];
    y = uv[1];
    r2 = x*x + y*y;
    r4 = r2*r2;
    r6 = r4*r2;
    a1 = 2*x*y;
    a2 = r2 + 2*x*x;
    a3 = r2 + 2*y*y;
    cdist = 1 + d_[0] * r2 + d_[1] * r4 + d_[4] * r6;
    xd = x * cdist + d_[2] * a1 + d_[3] * a2;
    yd = y * cdist + d_[2] * a3 + d_[3] * a1;
    px[0] = fx_ * xd + cx_;
    px[1] = fy_ * yd + cy_;
  }
  return px;
}

Vector2d Pinhole::world2cam(const Vector3d &xyz) const {
  return world2cam(utils::project2d(xyz));
}

} // namespace df
