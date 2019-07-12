#ifndef _DEPTH_FILTER_H_
#define _DEPTH_FILTER_H_


#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Core>

namespace depth_filter {

/// Each seed corresponds to a pixel in the reference keyframe
struct Seed {
  float a_;       // beta distribution parameter `a`
  float b_;       // beta distribution parameter `b` 
  float mu_;      // Mean of normal distribution
  float sigma2_;  // Variance of normal distribution
  cv::Mat patch_; // Patch around the seed in reference frame
  Seed() =default;
}; // class Seed

/// This implements sequential Bayesian updates for depth 
class DepthFilter {

DepthFilter() =default;


}; // class DepthFilter

} // namespace depth_filter

#endif

