#include "depth_filter/io.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <fstream>

namespace io {

IO::IO(std::string base) {
  const auto times_file = base + "/times.txt";
  const auto img_paths = base + "/image_0";
  const auto pose_file = base + "/pose.txt";

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

bool IO::read_set(size_t idx, double &ts, cv::Mat &img, Sophus::SE3 &T_w_f) {
  ts = times_[idx];
  img = cv::imread(img_paths_[idx], 0);
  T_w_f = poses_[idx];
  return true;
}

} // namespace io

