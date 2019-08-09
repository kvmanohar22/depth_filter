#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace io {

/// Loads images and ground truth 6-DOF poses
class IO {
public:
  IO(std::string env);
  bool read_set(size_t idx, double &ts, cv::Mat &img, Eigen::Matrix4d &T_f_w); 
  size_t n_imgs() const { return img_paths_.size(); }

private:
  std::vector<std::string> img_paths_;
  std::vector<Eigen::Matrix4d> poses_; 
  std::vector<double> times_; 

  
};

} // namespace io

#endif

