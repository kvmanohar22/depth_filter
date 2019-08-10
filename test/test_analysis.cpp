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
  void load_poses();
  inline void load_pose(size_t idx, Sophus::SE3 &T_w_f) { T_w_f = poses_[idx]; }

private:
  string root_dir_;
  vector<Sophus::SE3> poses_;
  vector<double> ts_;
  AbstractCamera *camera_;
};

void DepthAnalysis::load_poses() {
  std::ifstream fimages;
  fimages.open(root_dir_+"/info/images.txt");
  double ts;
  while(!fimages.eof()) {
    string s;
    std::getline(fimages, s);
    if (s.empty())
      continue;
    istringstream ss(s);
    ss >> ts; ss >> ts;
    ts_.push_back(ts);
  }
  cout << "Found " << ts_.size() << " images\n";
  poses_.reserve(ts_.size());
  ifstream fgt;
  fgt.open(root_dir_+"/info/groundtruth.txt");
  for (size_t ii=0; ii<ts_.size(); ++ii) {
    double pose[7];
    string s;
    std::getline(fgt, s);
    istringstream ss(s);
    size_t num=0;
    if (s.empty())
      continue;
    double t;
    ss >> t;
    while (num < 7) {
      ss >> pose[num];
      ++num; 
    }
    Matrix3d R_w_f; Vector3d t_w_f;
    for (size_t i=0;i<3;++i)
      t_w_f(i) = pose[i];
    utils::quaternion_to_rotation_matrix(pose+3, R_w_f);
    Sophus::SE3 T_w_f(R_w_f, t_w_f);
    poses_.emplace_back(T_w_f);
  }
}

DepthAnalysis::DepthAnalysis() {
  camera_ = new df::Pinhole(480, 640, 329.115520046, 329.115520046, 320.0, 240.0);
  root_dir_= std::getenv("DATA_RPG_SYNTHETIC");
  load_poses();
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

  // detect features in ref image
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  vector<cv::KeyPoint> kps_ref;
  cv::Mat des_ref;
  orb->detectAndCompute(img_ref, cv::noArray(), kps_ref, des_ref); 

#ifdef DEBUG_YES
  std::for_each(kps_ref.begin(), kps_ref.end(), [&](cv::KeyPoint &kpt) {
    cv::circle(img_ref, kpt.pt, 2, cv::Scalar(255, 0, 0));
  });
  cv::imshow("img_ref", img_ref);
  cv::waitKey(0);
#endif

  // load ground truth poses
  Sophus::SE3 T_w_ref, T_w_cur; 
  load_pose(0, T_w_ref);
  load_pose(1, T_w_cur);
}

}

int main() {
  df::DepthAnalysis analyzer;
  analyzer.run_two_view();
}
