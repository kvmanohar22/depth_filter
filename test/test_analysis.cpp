//
// Created by kv on 11/08/19.
//

#include "depth_filter/global.hpp"
#include "depth_filter/utils.h"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"

using namespace std;

namespace df {

class DepthAnalysis {
public:
  DepthAnalysis();
  ~DepthAnalysis() { delete camera_; }
  void run_two_view();
  void run_N_view();

private:
  string root_dir_;
  AbstractCamera *camera_;
};

DepthAnalysis::DepthAnalysis() {
    camera_ = new df::Pinhole(480, 640, 329.115520046, 329.115520046, 320.0, 240.0);
    root_dir_= std::getenv("DATA_RPG_SYNTHETIC");
  }

void DepthAnalysis::run_two_view() {
  string img_ref_file = root_dir_+"/img/img0001_0.png"; 
  string img_cur_file = root_dir_+"/img/img0002_0.png"; 

  cv::Mat img_ref = cv::imread(img_ref_file); 
  cv::Mat img_cur = cv::imread(img_cur_file); 
 
  cv::Mat depth_img_ref, depth_img_cur; 
  string depth_ref_file=root_dir_+"/depth/img0001_0.depth";
  string depth_cur_file=root_dir_+"/depth/img0002_0.depth";
  utils::load_blender_depth(depth_ref_file, *camera_, depth_img_ref); 
  utils::load_blender_depth(depth_cur_file, *camera_, depth_img_cur); 
}

}

int main() {
  df::DepthAnalysis analyzer;
  analyzer.run_two_view();
}
