#ifndef _DF_CORNER_H_
#define _DF_CORNER_H_

#include "depth_filter/global.hpp"
#include "depth_filter/point.h"
#include "depth_filter/frame.hpp"

namespace df {

class Point;

/// Corner in an image
class Corner {
private:
  Vector2d px_;    // pixel in the original image at level 0
  Vector3d f_;     // unit vector along the ray
  Point*   xyz_;   // 3D point in the world coordinates 
  FramePtr frame_; // frame in which this corner was first observed in

public:
  Corner() =default;
  Corner(Vector2d& px, FramePtr& frame)
    : px_(px),
      f_(frame_->cam_->cam2world(px)),
      xyz_(nullptr),
      frame_(frame)
  {}

  inline void add_point(Point* xyz) { xyz_ = xyz; }

  inline bool is_valid() { return xyz_ != nullptr; }

}; // class Corner

} // namespace df

#endif
