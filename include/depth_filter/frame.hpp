#ifndef _DEPTH_FILTER_FEATURES_HPP_
#define _DEPTH_FILTER_FEATURES_HPP_

#include "depth_filter/global.hpp"
#include "depth_filter/cameras/abstract.hpp"

namespace df {

class Corner;

class Frame {
public:
  AbstractCamera*   cam_;         // camera
  Sophus::SE3       T_f_w_;       // (w)orld -> (f)rame
  double            ts_;          // Timestamp of when the image was acquired
  long unsigned int idx_;         // Unique frame index
  list<Corner*>     fts_;         // list of corners in the image (only valid corners)

  Frame() =default;
 ~Frame() =default;
  Frame(Frame &frame);
  Frame(long unsigned int idx,
        const cv::Mat &img, AbstractCamera *cam,
        double ts);

  /// Camera instance
  inline AbstractCamera* cam() { return cam_; }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return -T_f_w_.rotation_matrix().transpose()*T_f_w_.translation(); }

  /// Set poses
  void set_pose(Sophus::SE3& T_w_f);
}; // class Frame

} // namespace df

#endif
