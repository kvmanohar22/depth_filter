#ifndef DEPTH_ESTIMATION_UTILS_H
#define DEPTH_ESTIMATION_UTILS_H

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

} // namespace utils

} // namespace df

#endif //DEPTH_ESTIMATION_UTILS_H
