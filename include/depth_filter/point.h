#ifndef _DF_POINT_H_
#define _DF_POINT_H_

#include "depth_filter/global.hpp"

namespace df {

class Corner;

class Point {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Point() =default;
  Point(Vector3d& xyz, Corner* corner)
    : xyz_(xyz) {
    obs_.push_back(corner);
  }

  // Add a new observation
  inline void add_observation(Corner* corner) { obs_.push_back(corner); } 

  // Frame which observers this point
  inline Corner* first_obs() const { return obs_.front(); }

private:
  Vector3d xyz_;
  list<Corner*> obs_;
}; // class Point

} // namespace df

#endif

