#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <string>

#include "depth_filter/global.hpp"
#include <opencv2/opencv.hpp>

namespace io {

/// Loads images and ground truth 6-DOF poses
class IO {
public:
  IO(std::string env);
  bool read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_f_w);
  size_t n_imgs() const { return img_paths_.size(); }

private:
  std::vector<std::string> img_paths_;
  std::vector<Sophus::SE3> poses_;
  Sophus::SE3              T_cam0_vel_; ///<! Velodyne to cam0
  std::vector<double>      times_;
};

} // namespace io

#endif

