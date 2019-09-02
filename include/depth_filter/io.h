#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <string>

#include "depth_filter/global.hpp"
#include "depth_filter/cloud.h"
#include <opencv2/opencv.hpp>

namespace io {

/// Loads images and ground truth 6-DOF poses
class IO {
public:
  IO() =default;
  virtual bool read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_f_w) const =0;
  inline size_t n_imgs() const { return img_paths_.size(); }

protected:
  std::vector<std::string> img_paths_;
  std::vector<Sophus::SE3> poses_;
  std::vector<double>      times_;
};

class Kitti : public IO {
public:
  Kitti(std::string env);
  virtual ~Kitti() =default;
  bool read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_f_w) const override;
  bool read_vel(size_t idx, df::PointCloud *cloud);
  Sophus::SE3 T_cam0_vel() { return T_cam0_vel_; }

private:
  Sophus::SE3              T_cam0_vel_; ///<! Velodyne to cam0
  std::vector<std::string> vel_paths_;
};

} // namespace io

#endif
