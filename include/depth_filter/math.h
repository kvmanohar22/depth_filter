#ifndef _DF_MATH_H
#define _DF_MATH_H

#include "depth_filter/global.h"

namespace utils {
namespace math {

inline Eigen::Matrix3f rotationX(float theta) {
  theta *= df::PI / 180.0f;
  float c = cos(theta);
  float s = sin(theta);
  Eigen::Matrix3f R;
  R << 1, 0, 0,
      0, c, -s,
      0, s, c;
  return R;
}

inline Eigen::Matrix3f rotationY(float theta)
{
  theta *= df::PI / 180.0f;
  float c = cos(theta);
  float s = sin(theta);
  Eigen::Matrix3f R;
  R << c, 0, s,
      0, 1, 0,
      -s, 0, c;
  return R;
}

inline Eigen::Matrix3f rotationZ(float theta)
{
  theta *= df::PI / 180.0f;
  float c = cos(theta);
  float s = sin(theta);
  Eigen::Matrix3f R;
  R << c, -s, 0,
      s, c, 0,
      0, 0, 1;
  return R;
}

} // namespace math
} // namespace utils

#endif