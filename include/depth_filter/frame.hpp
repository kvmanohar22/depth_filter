#ifndef _DEPTH_FILTER_FEATURES_HPP_
#define _DEPTH_FILTER_FEATURES_HPP_

#include "depth_filter/global.hpp"
#include "depth_filter/cameras/abstract.hpp"

namespace df {

class Frame {
public:
  AbstractCamera*   cam_;         // camera
  Matrix3d          R_f_w_;       // Rotation matrix from world to frame
  Vector3d          t_f_w_;       // translation vector from world to frame
  double            ts_;          // Timestamp of when the image was acquired
  long unsigned int idx_;         // Unique frame index

  Frame() =default;
  ~Frame() =default;
  Frame(Frame &frame);
  Frame(long unsigned int idx,
       const cv::Mat &img, AbstractCamera *cam,
       double ts);

  /// Camera instance
  inline AbstractCamera* cam() { return cam_; }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return -R_f_w_.transpose()*t_f_w_; }

  /// Set poses
  void set_pose(Matrix4d &T_w_f);
}; // class Frame

} // namespace df

#endif
