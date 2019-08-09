#include <iostream>

#include "depth_filter/io.h"

using namespace io;
using namespace std;
int main() {

  string base = std::getenv("DATA_KITTI");
  IO io(base+"/00");
  cout << "Read " << io.n_imgs() << " files\n"; 
 
  const auto n_files = io.n_imgs(); 
  for (size_t ii = 0; ii < n_files; ++ii) {
    cv::Mat img;
    double ts;
    Eigen::Matrix4d T_f_w;
    io.read_set(ii, ts, img, T_f_w);
  }
}

