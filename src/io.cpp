#include "depth_filter/global.hpp"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <fstream>

namespace io {

Kitti::Kitti(std::string base) {
  const auto times_file = base + "/times.txt";
  const auto img_paths = base + "/image_0";
  const auto vel_paths = base + "/velodyne";
  const auto pose_file = base + "/pose.txt";
  const auto calib_file = base + "/calib.txt";

  // Read image paths
  std::ifstream ftimes;
  ftimes.open(times_file.c_str());
  while(!ftimes.eof()) {
    std::string s;
    std::getline(ftimes, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      double t;
      ss >> t;
      times_.push_back(t);
    }
  }

  // Read image paths
  img_paths_.reserve(times_.size());
  for (int i = 0; i < times_.size(); ++i) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << i;
    img_paths_.emplace_back(img_paths + "/" + ss.str() + ".png");
    vel_paths_.emplace_back(vel_paths + "/" + ss.str() + ".bin");
  }

  // Read Transformation matrix
  std::ifstream fcalib;
  fcalib.open(calib_file.c_str());
  while(!fcalib.eof()) {
    std::string s;
    std::getline(fcalib, s);
    std::getline(fcalib, s);
    std::getline(fcalib, s);
    std::getline(fcalib, s);
    std::getline(fcalib, s);
    std::istringstream ss(s);
    std::string dummy;
    ss >> dummy;
    size_t num=0;
    Eigen::Matrix4d T_cam_vel = Eigen::Matrix4d::Identity();
    while (num < 12) {
      double t;
      ss >> t;
      T_cam_vel(num/4, num%4) = t;
      ++num;
    }
    Eigen::Matrix3d R_cam_vel; Eigen::Vector3d t_cam_vel;
    for (size_t i=0; i<3; ++i)
      for (size_t j=0; j<3; ++j)
        R_cam_vel(i, j) = T_cam_vel(i, j);
    for (size_t j=0; j<3; ++j)
      t_cam_vel(j) = T_cam_vel(j, 3);
    T_cam0_vel_ = Sophus::SE3(R_cam_vel, t_cam_vel);
    break;
  }

  // Read poses 
  poses_.reserve(times_.size());
  std::ifstream fposes;
  fposes.open(pose_file.c_str());
  while (!fposes.eof()) {
    std::string s;
    std::getline(fposes, s);
    Eigen::Matrix4d T_w_f = Eigen::Matrix4d::Identity();
    if (!s.empty()) {
      size_t num = 0;
      std::istringstream ss(s);
      while (num < 12) {
        double t;
        ss >> t;
        T_w_f(num/4, num % 4) = t;
        ++num;
      }
    }
    Eigen::Matrix3d R_w_f; Eigen::Vector3d t_w_f;
    for (size_t i=0; i<3; ++i)
      for (size_t j=0; j<3; ++j)
        R_w_f(i, j) = T_w_f(i, j);
    for (size_t j=0; j<3; ++j)
      t_w_f(j) = T_w_f(j, 3);
    Sophus::SE3 T_w_f_se3(R_w_f, t_w_f);
    poses_.emplace_back(T_w_f_se3);
  }
}

bool Kitti::read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_w_f) const {
  ts = times_[idx];
  img = cv::imread(img_paths_[idx], 0);
  T_w_f = poses_[idx];
  return true;
}

bool Kitti::read_vel(size_t idx, df::PointCloud *cloud) {
  df::utils::load_kitti_velodyne_scan(vel_paths_[idx], cloud);
}

Euroc::Euroc(std::string base, std::string hall_num)
  : hall_num_(hall_num) 
{
  base = base + "/" + hall_num_ + "/mav0";
  const auto pose_file  = base + "/pose.txt";
  const auto img_paths = base + "/cam0/data";

  // Read image paths
  std::ifstream ftimes;
  ftimes.open(times_file.c_str());
  while(!ftimes.eof()) {
    std::string s;
    std::getline(ftimes, s);
    if (!s.empty()) {
      double ts;
      std::string imgfile;
      std::istringstream ss(s);
      ss >> ts;
      ss >> imgfile;
      times_.push_back(ts);
      img_paths_.push_back(img_paths + "/" + imgfile);
    }
  }

  // Read poses
  poses_.reserve(times_.size());
  std::ifstream fposes;
  fposes.open(pose_file.c_str());
  double dummy;
  for (size_t ii=0; ii<times_.size(); ++ii) {
    std::string s;
    std::getline(fposes, s);
    Eigen::Matrix4d T_w_f = Eigen::Matrix4d::Identity();
    
    if (!s.empty()) {
      double pose[7];
      std::istringstream ss(s);
      ss >> dummy;
      size_t num = 0;
      while (num < 7) {
        ss >> pose[num];
        ++num; 
      }
    }
    Matrix3d R_w_f; Vector3d t_w_f;
    for (size_t i=0;i<3;++i)
      t_w_f(i) = pose[i];
    utils::quaternion_to_rotation_matrix(pose+3, R_w_f);
    Sophus::SE3 T_w_f(R_w_f, t_w_f);
    poses_.emplace_back(T_w_f);
  }
}

bool Euroc::read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_w_f) const {
  ts = times_[idx];
  img = cv::imread(img_paths_[idx], 0);
  T_w_f = poses_[idx];
  return true;
}

} // namespace io
