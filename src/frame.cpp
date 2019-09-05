#include "depth_filter/frame.h"

namespace df {

Frame::Frame(Frame &frame) :
        idx_(frame.idx_), cam_(nullptr), ts_(0) {}

Frame::Frame(const long unsigned int idx,
             cv::Mat &img, AbstractCamera *cam,
             double ts) :
        cam_(cam), idx_(idx), ts_(ts), img_(img.clone())
{
  assert(img.cols == cam->width() && img.rows == cam->height());
  create_img_pyramid(img);
}

void Frame::set_pose(Sophus::SE3& T_w_f) {
  T_f_w_ = T_w_f.inverse();
}

void Frame::create_img_pyramid(cv::Mat& src) {
  size_t n_lvls = 3; // change this?
  cv::Mat tmp = src;
  pyr_.push_back(src);
  for (size_t lvl=0; lvl<n_lvls; ++lvl) {
    cv::Mat dst;
    cv::pyrDown(tmp, dst, cv::Size(tmp.cols/2, tmp.rows/2));
    tmp = dst;
    pyr_.push_back(dst); 
  }
}

} // namespace df
