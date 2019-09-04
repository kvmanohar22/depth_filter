#ifndef _HAWK_VIEWER_HPP_
#define _HAWK_VIEWER_HPP_

#include "depth_filter/global.hpp"
#include <thread>
#include <pangolin/pangolin.h>

namespace df {

/// 3D Viewer
class Viewer3D {
private:
  /*
    Viewer axes convention (Right-handed system)
    x-axis: Right
    y-axis: Down
    z-axis: Front
  */
  float H, W;
  std::string window_name;

  pangolin::View d_cam;
  pangolin::OpenGlRenderState s_cam;

  // Run this in background
  std::thread render_loop_;
  bool finish_render_;

public:
   Viewer3D(vector<Vector3d>& points, vector<Vector3d>& colors, vector<Matrix4d>& cameras);
  ~Viewer3D();

  // setup render loop in a separater thread
  void setup(vector<Vector3d>& points, vector<Vector3d>& colors, vector<Matrix4d>& cameras);

  // render camera  
  void draw_camera(Matrix4d &Rt, cv::Point3f &color);

  // update after each frame
  void update(vector<Vector3d>& points, vector<Vector3d>& colors, vector<Matrix4d>& cameras);

  void check_axes();

  // Draw the axes
  void draw_axes();
}; // class Viewer3D

} // namespace df

#endif
