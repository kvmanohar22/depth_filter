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
  void update_seed(FramePtr& frame_cur, Seed& seed);

  struct Options {
    int max_frames_;
    SeedsInitType seeds_init_type_;
    Options()
      : max_frames_(20),
        seeds_init_type_(SeedsInitType::OUTSOURCE)
    {}
  } options_;



private:
  FramePtr        frame_ref_; // reference frame from which new seeds are initialized
  list<Seed>      seeds_;     // list of seeds from the reference frame
  queue<FramePtr> frames_;    // queue of frames from which seeds are updated

}; // class DepthFilter

} // namespace df

#endif

