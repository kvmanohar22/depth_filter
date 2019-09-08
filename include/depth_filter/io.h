#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <string>

#include "depth_filter/global.h"
#include "depth_filter/cloud.h"
#include "depth_filter/utils.h"
#include "depth_filter/cameras/abstract.h"
#include <opencv2/opencv.hpp>

namespace io {

/// Loads images and ground truth 6-DOF poses
class IO {
public:
  IO() =default;
  virtual ~IO() =default;
  bool read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_f_w) const;
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
  inline bool read_vel(size_t idx, df::PointCloud *cloud) {
    return df::utils::load_kitti_velodyne_scan(vel_paths_[idx], cloud);
  }
  Sophus::SE3 T_cam0_vel() { return T_cam0_vel_; }

private:
  Sophus::SE3              T_cam0_vel_; ///<! Velodyne to cam0
  std::vector<std::string> vel_paths_;
};

class RPGSyntheticForward : public IO {
public:
  RPGSyntheticForward(std::string env);
  inline bool read_gtdepth(size_t idx, df::AbstractCamera* camera, cv::Mat& depth_img_ref_, df::utils::DepthType type) {
    df::utils::load_blender_depth(gt_depth_paths_[idx], *camera, depth_img_ref_, type);
    return true;
  }
  virtual ~RPGSyntheticForward() =default;

private:
  std::vector<std::string> gt_depth_paths_;
};

class RPGSyntheticDownward : public IO {
public:
  RPGSyntheticDownward(std::string env);
  virtual ~RPGSyntheticDownward() =default;
  inline bool read_gtdepth(size_t idx, df::AbstractCamera* camera, cv::Mat& depth_img_ref_, df::utils::DepthType type) {
    df::utils::load_blender_depth(gt_depth_paths_[idx], *camera, depth_img_ref_, type);
    return true;
  }

private:
  std::vector<std::string> gt_depth_paths_;
};

} // namespace io

#endif
