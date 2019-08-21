#include <iostream>

#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/cloud.h"

using namespace io;
using namespace std;

void test() {
  string base = std::getenv("DATA_KITTI");
  auto file = base + "/00/velodyne/000000.bin";
  df::PointCloud cloud;

  if(!df::utils::load_kitti_velodyne_scan(file, &cloud))
    return;

  cout << "Read " << cloud.npts() << " points" << endl;
  cout << cloud.points_[cloud.npts()-1].transpose() << endl;
}

int main() {
  test();
}
