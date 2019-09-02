#include "depth_filter/frame.hpp"

namespace df {

Frame::Frame(Frame &frame) :
        idx_(frame.idx_), cam_(nullptr), ts_(0) {}

Frame::Frame(const long unsigned int idx,
             const cv::Mat &img, AbstractCamera *cam,
             double ts) :
        cam_(cam), idx_(idx), ts_(ts), img_(img.clone()) {}

void Frame::set_pose(Sophus::SE3& T_w_f) {
  T_f_w_ = T_w_f.inverse();
}

} // namespace df
