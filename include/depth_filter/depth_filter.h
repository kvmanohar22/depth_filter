#ifndef _DEPTH_FILTER_H_
#define _DEPTH_FILTER_H_

#include "depth_filter/global.h"
#include "depth_filter/frame.h"
#include "depth_filter/seed.h"

#include "vikit/patch_score.h"
#include "vikit/vision.h"

#include <opencv2/opencv.hpp>

namespace df {

enum class SeedsInitType {
  FAST_DETECTOR,
  OUTSOURCE  
};

typedef vk::patch_score::ZMSSD<4> PatchScore;

/// This implements sequential Bayesian updates for depth 
class DepthFilter {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DepthFilter(AbstractCamera *cam);

  // Add a new keyframe. New seeds are initialized from this
  void add_keyframe(FramePtr& frame, float depth_min=0.0f, float depth_mean=0.0f, float depth_max=0.0f);

  // Add a new frame to update the seeds
  void add_frame(FramePtr& frame);

  // Seeds are initialized from the current keyframe 
  void initialize_seeds(float depth_min, float depth_mean, float depth_max);

  // Seeds are initialized from external source
  list<Seed>& get_mutable_seeds();

  // Seeds are initialized from external source
  inline const list<Seed>& get_seeds() { return seeds_; }

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
        Corner* corner, float d_current, float d_min, float d_max, float &d_new);

  // compute variance in triangulation
  float compute_tau(Vector3d rp, Vector3d t, Vector3d f);

  void create_ref_patch(uint8_t* patch_ref, Vector2d &px_ref, cv::Mat &img);

  bool triangulate(const Sophus::SE3& T_search_ref, const Vector3d& f_ref,
    const Vector3d& f_cur, float& depth);

  struct Options {
    int max_frames_;
    SeedsInitType seeds_init_type_;
    float sigma2_convergence_thresh_;
    size_t patch_size_;
    size_t half_patch_size_;
    bool verbose_;
    size_t max_steps_;
    Options()
      : max_frames_(20),
        seeds_init_type_(SeedsInitType::OUTSOURCE),
        sigma2_convergence_thresh_(200.0f),
        patch_size_(8),
        half_patch_size_(4),
        verbose_(false),
        max_steps_(500)
    {}
  } options_;

private:
  FramePtr        frame_ref_; // reference frame from which new seeds are initialized
  list<Seed>      seeds_;     // list of seeds from the reference frame
  queue<FramePtr> frames_;    // queue of frames from which seeds are updated
  float           one_px_angle_;

}; // class DepthFilter

} // namespace df

#endif

