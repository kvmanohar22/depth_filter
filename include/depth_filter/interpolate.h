#ifndef _DF_INTERPOLATE_H_
#define _DF_INTERPOLATE_H_

#include "depth_filter/global.h"

namespace utils {
namespace interpolate {

template <typename T>
T interpolate(cv::Mat& img, Eigen::Vector2d& px) {
  T val=0;
  int x_int = floor(px.x());
  int y_int = floor(px.y());

  double wcl = x_int-px.x();
  double wcd = x_int-px.x();
  double wcu = 1.0 - wcd;
  double wcr = 1.0 - wcl;

  T vcl = img.at<T>(y_int, x_int);
  T vcd = img.at<T>(y_int+1, x_int);
  T vcr = img.at<T>(y_int, x_int+1);
  T vcu = img.at<T>(y_int-1, x_int);

  return vcl*wcl+vcd*wcd+vcr*wcr+vcu*wcu;
}

} // namespace interpolate
} // namespace utils

#endif

