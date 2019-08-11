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
  string img_cur_file = root_dir_+"/img/img0005_0.png"; 

  cv::Mat img_ref = cv::imread(img_ref_file); 
  cv::Mat img_cur = cv::imread(img_cur_file); 
 
  cv::Mat depth_img_ref, depth_img_cur; 
  string depth_ref_file=root_dir_+"/depth/img0001_0.depth";
  string depth_cur_file=root_dir_+"/depth/img0005_0.depth";
  utils::load_blender_depth(depth_ref_file, *camera_, depth_img_ref); 
  utils::load_blender_depth(depth_cur_file, *camera_, depth_img_cur); 

  // detect features in ref image
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  vector<cv::KeyPoint> kps_ref;
  cv::Mat des_ref;
  orb->detectAndCompute(img_ref, cv::noArray(), kps_ref, des_ref); 

#ifdef DEBUG_YES
  auto temp_img = img_ref.clone();
  std::for_each(kps_ref.begin(), kps_ref.end(), [&](cv::KeyPoint &kpt) {
    cv::circle(temp_img, kpt.pt, 2, cv::Scalar(255, 0, 0));
  });
  cv::imshow("img_ref", temp_img);
  cv::waitKey(3000);
#endif

  // load ground truth poses
  Sophus::SE3 T_w_ref, T_w_cur, T_cur_ref;
  load_pose(0, T_w_ref);
  load_pose(4, T_w_cur);
  T_cur_ref = T_w_cur.inverse() * T_w_ref;

  srand(time(0));
  size_t rand_idx = rand()%kps_ref.size();
  auto kpt = kps_ref[rand_idx].pt;
  auto d = depth_img_ref.at<float>(kpt.y, kpt.x);

  double dz = 1.0;
  double d_min = 0.1*d;
  double d_max = 10*d;
  Vector2d px(kpt.x, kpt.y);
  Vector3d bearing_vec_ref = camera_->cam2world(px);

#ifdef DEBUG_YES
  auto t_img_ref = img_ref.clone();
  auto t_img_cur = img_cur.clone();
  Vector3d true_xyz = bearing_vec_ref*d;
  auto px_cur = camera_->world2cam(T_cur_ref * true_xyz);
  cv::circle(t_img_ref, kpt, 1, cv::Scalar(255, 0, 0), CV_AA);
  cv::circle(t_img_cur, cv::Point2f(px_cur.x(), px_cur.y()), 1, cv::Scalar(255, 0, 0), CV_AA);
  cv::imshow("img_ref [true depth]", t_img_ref);
  cv::imshow("img_cur [true depth]", t_img_cur);
  cv::waitKey(3000);
#endif

  while (d_min < d_max) {
    cout << "original depth = " << d << "\td_max = " << d_max << "\t" << "d_cur = " << d_min << endl;
    Vector3d pt_ref = bearing_vec_ref * d_min;
    Vector3d pt_cur = T_cur_ref * pt_ref;
    Vector2d uv_cur = camera_->world2cam(pt_cur);

    cv::Mat img_ref_n = img_ref.clone();
    cv::Mat img_cur_n = img_cur.clone();
    cv::circle(img_ref_n, kps_ref[rand_idx].pt, 2, cv::Scalar(255, 0, 0), CV_AA);
    cv::circle(img_cur_n, cv::Point2f(uv_cur.x(), uv_cur.y()), 2, cv::Scalar(0, 255, 0), CV_AA);

    cv::imshow("img_ref", img_ref_n);
    cv::imshow("img_cur", img_cur_n);
    cv::waitKey(0);
    d_min += dz;
  }
}

}

int main() {
  df::DepthAnalysis analyzer;
  analyzer.run_two_view();
}
