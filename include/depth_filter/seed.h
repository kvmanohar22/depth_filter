#ifndef _DEPTH_FILTER_SEED_H_
#define _DEPTH_FILTER_SEED_H_

#include "depth_filter/global.hpp"
#include <opencv2/opencv.hpp>

namespace df {

/// Each seed corresponds to a pixel in the reference keyframe
struct Seed {
  float a_;       // beta distribution parameter `a`
  float b_;       // beta distribution parameter `b` 
  float mu_;      // Mean of normal distribution
  float sigma2_;  // Variance of normal distribution
  cv::Mat patch_; // Patch around the seed in reference frame
  Seed() =default;
}; // class Seed

} // namespace df

#endif

