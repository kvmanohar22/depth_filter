#include "depth_filter/global.hpp"
#include "depth_filter/utils.h"

using namespace std;

int main() {
  double q[] = {0, 0, 0, 1};
  Eigen::Matrix3d R;
  df::utils::quaternion_to_rotation_matrix(q, R);
  for (size_t i=0;i<3;++i) {
    for (size_t j=0;j<3;++j) {
      cout << R(i, j) << "\t";
    }
    cout << endl;
  }
}