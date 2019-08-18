#include "depth_filter/global.hpp"
#include "depth_filter/utils.h"

using namespace std;

void test_quaternion() {
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

void test_cc() {
  // case 1
  cv::Mat f(4, 4, CV_32F), g(4,4,CV_32F);
  f = (cv::Mat_<float>(4, 4) << 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7);
  g = (cv::Mat_<float>(4, 4) << 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7);
  f /= 10;
  g /= 10;
  // cv::imshow("f", f);
  // cv::waitKey(0);
  // auto score = df::utils::normalized_cross_correlation(f, g);
  // cv::imshow("f", f);
  // cv::waitKey(0);

  // case 2
  cv::Mat f1 = cv::imread("../data/test_image.png", 0);
  cv::Mat g1 = cv::imread("../data/test_image_patch.png", 0);
  if(f1.empty() || g1.empty())
    throw std::runtime_error("img is empty\n");
  f1.convertTo(f1, CV_32F);
  g1.convertTo(g1, CV_32F);

  cv::imwrite("f1_before.png", f1);
  auto score = df::utils::normalized_cross_correlation(f1, g1);
  cv::imwrite("f1_after.png", f1);
}


int main() {
  // test_quaternion();
  test_cc();
}
