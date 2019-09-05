#ifndef _DEPTH_FILTER_SEED_H_
#define _DEPTH_FILTER_SEED_H_

#include "depth_filter/global.h"
#include "depth_filter/corner.h"
#include <opencv2/opencv.hpp>

namespace df {

/// Each seed corresponds to a pixel in the reference keyframe
class Seed {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Seed(float depth_mean, float depth_min, float depth_max, Corner *corner)
  : idx_(seed_counter++),
    a_(10),
    b_(10),
    mu_(depth_mean),
    z_range_(depth_max - depth_min),
    sigma2_(z_range_*z_range_/36), // TODO: Change this?
    corner_(corner)
  {}

  static size_t seed_counter;
  size_t idx_;     // unique index of seed
  float a_;        // beta distribution parameter `a`
  float b_;        // beta distribution parameter `b` 
  float mu_;       // Mean of normal distribution
  float z_range_;  // range in which depth is considered
  float sigma2_;   // Variance of normal distribution
  cv::Mat patch_;  // Patch around the seed in reference frame
  Corner* corner_; // corner from which this seed is initialized
}; // class Seed

/// Each seed corresponds to a pixel in the reference keyframe
class ConvergedSeed : public Seed {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  ConvergedSeed();

private:
  float d_hat_;   // true depth
  float error_;   // L2 distance b/w true and state depth
}; // class ConvergedSeed

} // namespace df

#endif

