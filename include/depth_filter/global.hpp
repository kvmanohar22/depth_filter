#ifndef _DEPTH_FILTER_GLOBAL_HPP_
#define _DEPTH_FILTER_GLOBAL_HPP_

#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <random>

#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/StdVector>
#include <Eigen/src/Core/Matrix.h>

#include <glog/logging.h>

#include <sophus/se3.h>

namespace df {

  using namespace std;
  using namespace Eigen;

  class Frame;
  typedef boost::shared_ptr<Frame> FramePtr;

  static const float EPS = 1e-7;
  static const double PI = 3.14159265359;
} // namespace df

#endif
