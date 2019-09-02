#include "depth_filter/depth_filter.h"

namespace df {

DepthFilter::DepthFilter() {
  LOG(INFO) << "DepthFilter initialized...";
}

void DepthFilter::add_keyframe(FramePtr& frame) {
  DLOG(INFO) << "New KeyFrame added\n";
  frame_ref_ = frame;
  if (options_.seeds_init_type_ == SeedsInitType::FAST_DETECTOR)
    initialize_seeds();
}

void DepthFilter::add_frame(FramePtr& frame) {
  if (frames_.size() > options_.max_frames_)
    frames_.pop();
  frames_.push(frame);
}

void DepthFilter::initialize_seeds() {

}

list<Seed>& DepthFilter::get_mutable_seeds() {
  return seeds_;
}

void DepthFilter::update_seeds_loop() {
  /// main outer loop for updating seeds
  while (true) {

  }
}

void DepthFilter::update_seed(FramePtr& frame, Seed& seed) {

}


} // namespace df

