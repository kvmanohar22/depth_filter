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

void test_cross_correlation_single_patch() {
  cv::Mat f, g;
  f = (cv::Mat_<float>(4, 4) << 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7);
  g = (cv::Mat_<float>(4, 4) << 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7);
  f = df::utils::normalize_image(f);
  g = df::utils::normalize_image(g);
  auto score = df::utils::cross_correlation_single_patch(
      f.ptr<float>(), g.ptr<float>(), 4, 4, 0, 0, 0, 0, 4, 4);
  cout << "NCC score = " << score << endl;

  f = (cv::Mat_<float>(3, 5) << 1,2,3,4,5,6,7,8,9,1,2,3,4,5,7);
  g = (cv::Mat_<float>(3, 5) << 1,2,3,4,5,6,7,8,9,1,2,3,4,5,7);
  f = df::utils::normalize_image(f);
  g = df::utils::normalize_image(g);
  score = df::utils::cross_correlation_single_patch(
      f.ptr<float>(), g.ptr<float>(), 5, 5, 0, 0, 0, 0, 5, 3);
  cout << "NCC score = " << score << endl;
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
  // df::utils::normalized_cross_correlation(f, g);
  // cv::imshow("f", f);
  // cv::waitKey(0);

  // case 2
  cv::Mat f1 = cv::imread("../data/test_image.png", 0);
  cv::Mat g1 = cv::imread("../data/test_image_patch.png", 0);
  if(f1.empty() || g1.empty())
    throw std::runtime_error("img is empty\n");
  f1.convertTo(f1, CV_32F);
  g1.convertTo(g1, CV_32F);

  auto f1_normalized = df::utils::normalize_image(f1, 0, 0, 4, 4);
  auto g1_normalized = df::utils::normalize_image(g1);
  float *f1_ptr = f1_normalized.ptr<float>();
  float *g1_ptr = g1_normalized.ptr<float>();
  float ncc_score = df::utils::cross_correlation_single_patch(
      f1_ptr, f1_ptr, f1.cols, f1.cols,
      0, 0, 0, 0, 4, 4);
  cout << "NCC = " << ncc_score << endl;
}


int main() {
  test_quaternion();
  test_cc();
  test_cross_correlation_single_patch();
}
