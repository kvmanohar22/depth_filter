#ifndef _DEPTH_FILTER_ABSTRACT_CAMERA_HPP_
#define _DEPTH_FILTER_ABSTRACT_CAMERA_HPP_

#include "depth_filter/global.hpp"

namespace df {

class AbstractCamera {
protected:
  int width_;
  int height_;

public:
  AbstractCamera() : width_(0), height_(0) {}
  AbstractCamera(int width, int height) :
    width_(width), height_(height) {}

  virtual ~AbstractCamera() =default;

  virtual Vector3d cam2world(const double &u, const double &v) const =0;
  virtual Vector3d cam2world(const Vector2d &px) const =0;

  virtual Vector2d world2cam(const Vector3d &xyz) const =0;
  virtual Vector2d world2cam(const Vector2d &uv) const =0;

  virtual double error2() const =0;

  inline int width()  { return width_;  }
  inline int height() { return height_; }

  inline bool is_in_frame(const Vector2i &obs, int boundary=0) const {
    if (obs[0] >= boundary && obs[0] < width_ - boundary &&
      obs[1] >= boundary && obs[1] < height_ - boundary)
      return true;
    return false;
  }
};

} // namespace df

#endif
