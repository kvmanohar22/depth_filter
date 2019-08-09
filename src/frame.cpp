#include "depth_filter/frame.hpp"

namespace df {

Frame::Frame(Frame &frame) :
        idx_(frame.idx_), cam_(nullptr), ts_(0) {}

Frame::Frame(const long unsigned int idx,
             const cv::Mat &img, AbstractCamera *cam,
             double ts) :
        cam_(cam), idx_(idx), ts_(ts) {}

void Frame::set_pose(Matrix4d &T_w_f) {
  Matrix3d R; Vector3d t;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      R(i, j) = T_w_f(i, j);
  for (size_t i = 0; i < 3; ++i)
    t(i) = T_w_f(i, 3);
  R_f_w_ = R.transpose();
  t_f_w_ = -R_f_w_ * t;
}

} // namespace df
