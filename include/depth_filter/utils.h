#ifndef DEPTH_ESTIMATION_UTILS_H
#define DEPTH_ESTIMATION_UTILS_H

#include "depth_filter/global.hpp"
#include "depth_filter/cloud.h"
#include "depth_filter/cameras/abstract.hpp"

namespace df {

namespace utils {

inline Vector2d project2d(const Vector3d &v) {
  return v.head<2>()/v.z();
}

inline void load_blender_depth(
    const std::string file_name,
    const df::AbstractCamera& cam,
    cv::Mat& img) {
  std::ifstream file_stream(file_name.c_str());
  assert(file_stream.is_open());
  img = cv::Mat(cam.height(), cam.width(), CV_32FC1);
  float * img_ptr = img.ptr<float>();
  float depth;
  for(int y=0; y<cam.height(); ++y) {
    for(int x=0; x<cam.width(); ++x, ++img_ptr) {
      file_stream >> depth;
      Eigen::Vector2d uv(utils::project2d(cam.cam2world(x,y)));
      *img_ptr = depth * sqrt(uv[0]*uv[0] + uv[1]*uv[1] + 1.0);
      if(file_stream.peek() == '\n' && x != cam.width()-1 && y != cam.height()-1)
        printf("WARNING: did not read the full depthmap!\n");
    }
  }
}


// Ref: https://github.com/kashif/ceres-solver/blob/master/include/ceres/rotation.h
// q[0] -> i^th component 
// q[1] -> j^th component 
// q[2] -> k^th component 
// q[3] -> scalar component 
// R -> row-major order
inline void  quaternion_to_rotation_matrix(double q[4], Matrix3d &R) {
  double b = q[0];
  double c = q[1];
  double d = q[2];
  double a = q[3];

  double aa = a * a;
  double ab = a * b;
  double ac = a * c;
  double ad = a * d;
  double bb = b * b;
  double bc = b * c;
  double bd = b * d;
  double cc = c * c;
  double cd = c * d;
  double dd = d * d;

  R(0, 0) =  aa + bb - cc - dd; R(0, 1) = double(2) * (bc - ad);   R(0, 2) = double(2) * (ac + bd);
  R(1, 0) = double(2) * (ad + bc);   R(1, 1) = aa - bb + cc - dd;  R(1, 2) = double(2) * (cd - ab);
  R(2, 0) = double(2) * (bd - ac);   R(2, 1) = double(2) * (ab + cd);   R(2, 2) = aa - bb - cc + dd;  

  double normalizer = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
  assert(normalizer > double(0));
  normalizer = double(1) / normalizer;

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      R(i, j) *= normalizer;
}


bool cross_correlation(cv::Mat &f, cv::Mat &g);

// Note; f and g should be row-major ordered
// f -> (0, 0)
// g -> (0, 0)
// start_idx_fx -> along the width of the image
// tx -> template width (along x)
float cross_correlation_single_patch(float *f, float *g,
    size_t f_cols, size_t g_cols,
    size_t start_idx_fx, size_t start_idx_fy,
    size_t start_idx_gx, size_t start_idx_gy,
    size_t tx, size_t ty);

cv::Mat normalize_image(cv::Mat &img,
    size_t r_idx=0, size_t c_idx=0,
    int cols=-1, int rows=-1);

void normalized_cross_correlation(cv::Mat &f, cv::Mat &g);

bool load_kitti_velodyne_scan(std::string file, df::PointCloud *cloud);

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  T val = a;
  for (auto &itr : xs) {
      itr = val;
      val += h;
  }
  return xs;
}

} // namespace utils

} // namespace df

#endif //DEPTH_ESTIMATION_UTILS_H
