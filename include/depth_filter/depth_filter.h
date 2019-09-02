#ifndef _DEPTH_FILTER_H_
#define _DEPTH_FILTER_H_

#include "depth_filter/global.hpp"
#include "depth_filter/frame.hpp"
#include "depth_filter/seed.h"
#include <opencv2/opencv.hpp>

namespace df {

enum class SeedsInitType {
  FAST_DETECTOR,
  OUTSOURCE  
};

/// This implements sequential Bayesian updates for depth 
class DepthFilter {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DepthFilter();

  // Add a new keyframe. New seeds are initialized from this
  void add_keyframe(FramePtr& frame);

  // Add a new frame to update the seeds
  void add_frame(FramePtr& frame);

  // Seeds are initialized from the current keyframe 
  void initialize_seeds();

  // Seeds are initialized from external source
  list<Seed>& get_mutable_seeds();

  // Seeds upate infinite loop
  void update_seeds_loop();

  // Update a single seed
  void update_seed(Seed* seed, float mu, float tau2);

  // Update seeds from this frame
  void update_seeds(FramePtr& frame_cur);

  // TODO: Should make this thread safe
  inline size_t n_seeds() const { return seeds_.size(); }

  // Search for a patch along epipolar line with max. NCC score
  bool find_match_along_epipolar(FramePtr frame_ref, FramePtr frame_cur,
        Corner* corner, float d_current, float d_min, float d_max, float d_new);

  // compute variance in triangulation
  float compute_tau(Vector3d rp, Vector3d t, Vector3d f, float one_px_angle);

  // angle generated by 1 pixel
  float one_pixel_angle();


  struct Options {
    int max_frames_;
    SeedsInitType seeds_init_type_;
    float thresh_outlier_;
    float thresh_inlier_;
    float epsilon_;
    Options()
      : max_frames_(20),
        seeds_init_type_(SeedsInitType::OUTSOURCE),
        thresh_outlier_(0.05),                       // 
        thresh_inlier_(0.1)                          //
    {}
  } options_;

private:
  FramePtr        frame_ref_; // reference frame from which new seeds are initialized
  list<Seed>      seeds_;     // list of seeds from the reference frame
  queue<FramePtr> frames_;    // queue of frames from which seeds are updated

}; // class DepthFilter

} // namespace df

#endif

