#ifndef _DEPTH_FILTER_CLOUD_H_
#define _DEPTH_FILTER_CLOUD_H_

#include "depth_filter/global.hpp"

namespace df {

class PointCloud {
public:
  vector<Eigen::Vector3f> points_;

  PointCloud() {}
  ~PointCloud() {}
  inline Eigen::Vector3f point(size_t idx) const { return points_[idx]; }
  inline size_t npts() const { return points_.size(); }
};

}

#endif
