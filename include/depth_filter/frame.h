#ifndef _DEPTH_FILTER_FEATURES_HPP_
#define _DEPTH_FILTER_FEATURES_HPP_

#include "depth_filter/global.h"
#include "depth_filter/cameras/abstract.h"

namespace df {

class Corner;
typedef vector<cv::Mat> ImgPyr;

class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AbstractCamera*   cam_;         // camera
  Sophus::SE3       T_f_w_;       // (w)orld -> (f)rame
  double            ts_;          // Timestamp of when the image was acquired
  long unsigned int idx_;         // Unique frame index
  list<Corner*>     fts_;         // list of corners in the image (only valid corners)
  cv::Mat           img_;         // image
  ImgPyr            pyr_;         // image pyramid

  Frame() =default;
 ~Frame() =default;
  Frame(Frame &frame);
  Frame(long unsigned int idx,
        cv::Mat &img, AbstractCamera *cam,
        double ts);

  /// Camera instance
  inline AbstractCamera* cam() { return cam_; }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return -T_f_w_.rotation_matrix().transpose()*T_f_w_.translation(); }

  /// create image pyramid
  void create_img_pyramid(cv::Mat& img);

  /// Set poses
  void set_pose(Sophus::SE3& T_w_f);
}; // class Frame

} // namespace df

#endif
