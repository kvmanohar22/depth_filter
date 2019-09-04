#include "depth_filter/feature_detector.h"
#include "depth_filter/global.hpp"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"

int main() {
  /// Fast on single image 
  std::string imgfile("./../data/test_image.png");
  cv::Mat img = cv::imread(imgfile.c_str(), 0);
  assert(!img.empty());
  std::list<df::Corner*> corners; 
  df::FastDetector::detect(img, corners);
  cv::Mat dst = img.clone();
  cv::cvtColor(dst, dst, CV_GRAY2BGR);
  std::for_each(corners.begin(), corners.end(), [&](df::Corner* corner) {
    cv::circle(dst, cv::Point2f(corner->px_.x(), corner->px_.y()), 1, cv::Scalar(255, 0, 0), 1); 
    delete corner;
  }); 
  cv::imshow("single", dst);
  cv::waitKey(0);
}
