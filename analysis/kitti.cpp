#include <iostream>

#include "depth_filter/global.hpp"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"

using namespace io;
using namespace std;
using namespace df;

class DepthAnalysis {
public:
  DepthAnalysis();
  ~DepthAnalysis();
  void run_two_view();
  void run_N_view();

private:
  string root_dir_;
  AbstractCamera *camera_;
  IO *io_;
};

DepthAnalysis::DepthAnalysis() {
  camera_ = new df::Pinhole(1241, 376, 718.856, 718.856, 607.1928, 185.2157);
  root_dir_= std::getenv("DATA_KITTI");
  io_ = new IO(root_dir_+"/00");
  cout << "Read " << io_->n_imgs() << " files\n";
}

DepthAnalysis::~DepthAnalysis() {
  delete camera_;
  delete io_;
}

void DepthAnalysis::run_two_view() {
  cv::Mat img_ref, img_cur;
  double ts_ref, ts_cur;
  Sophus::SE3 T_ref_w, T_cur_w;
  io_->read_set(0, ts_ref, img_ref, T_ref_w);  
  io_->read_set(2, ts_cur, img_cur, T_cur_w);  

  cv::imshow("ref", img_ref);
  cv::waitKey(0);

}

int main() {
  DepthAnalysis analyzer;
  analyzer.run_two_view();
}

