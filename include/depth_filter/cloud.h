#ifndef _DEPTH_FILTER_CLOUD_H_
#define _DEPTH_FILTER_CLOUD_H_

#include "depth_filter/global.h"
#include "depth_filter/math.h"

namespace df {

class PointCloud {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  vector<Eigen::Vector3f> points_;

  PointCloud() {}
  ~PointCloud() {}
  inline Eigen::Vector3f point(size_t idx) const { return points_[idx]; }
  inline size_t npts() const { return points_.size(); }

  inline void translate(Vector3f& t) {
    std::for_each(points_.begin(), points_.end(), [&](Vector3f &pt) {
      pt += t;
    });
  }

  inline void rotateX(float theta) {
    auto R = utils::math::rotationX(theta);
    std::for_each(points_.begin(), points_.end(), [&](Vector3f &pt) {
      pt = R*pt;
    });
  }
  inline void rotateY(float theta) {
    auto R = utils::math::rotationY(theta);
    std::for_each(points_.begin(), points_.end(), [&](Vector3f &pt) {
      pt = R*pt;
    });
  }
  inline void rotateZ(float theta) {
    auto R = utils::math::rotationZ(theta);
    std::for_each(points_.begin(), points_.end(), [&](Vector3f &pt) {
      pt = R*pt;
    });
  }
};

}

#endif
