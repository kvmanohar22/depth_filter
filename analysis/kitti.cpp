#include <iostream>
#include <cstdio>

#include "depth_filter/global.hpp"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/cloud.h"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"

using namespace io;
using namespace std;
using namespace df;

static const int IMG_H = 376;
static const int IMG_W = 1241;
static const int patch_size = 4;
static const int half_patch_size = patch_size / 2;

static const int ref_idx = 27;
static const int cur_idx = 30;


bool valid(Vector2d &pt) {
  if (pt.x() < 0 || pt.y() < 0 ||
      pt.x() > IMG_W || pt.y() > IMG_H)
    return false;
  return true;
}

void draw_line(Vector3d &line, cv::Mat &img) {
  Vector3d th(0, 1, 0);
  Vector3d lv(1, 0, 0);
  Vector3d bh(0, 1, -img.rows);
  Vector3d rv(1, 0, -img.cols);

  auto pt_th = utils::project2d(line.cross(th));
  auto pt_lv = utils::project2d(line.cross(lv));
  auto pt_bh = utils::project2d(line.cross(bh));
  auto pt_rv = utils::project2d(line.cross(rv));
 
  vector<Vector2d> pts;

  if (valid(pt_th))
    pts.push_back(pt_th);
  if (valid(pt_lv))
    pts.push_back(pt_lv);
  if (valid(pt_bh))
    pts.push_back(pt_bh);
  if (valid(pt_rv))
    pts.push_back(pt_rv);
  
  if (pts.size() == 2)
    cv::line(img,
        cv::Point2f(pts.front().x(), pts.front().y()),
        cv::Point2f(pts.back().x(), pts.back().y()),
        cv::Scalar(255, 0, 0), 1, CV_AA);
}

class DepthAnalysis {
public:
  DepthAnalysis(size_t ref_idx);
  ~DepthAnalysis();
  void run_two_view(size_t cur_idx);
  void run_N_view();

protected:
  void load_ref();

private:
  string          root_dir_;
  AbstractCamera *camera_;
  IO             *io_;

  /// Reference image stuff
  size_t               ref_idx_;
  double               ref_ts_;
  cv::Mat              ref_img_;
  Sophus::SE3          T_w_ref_;
  PointCloud           cloud_;
  vector<cv::Point2f>  ref_kps_;
};

DepthAnalysis::DepthAnalysis(size_t ref_idx) : ref_idx_(ref_idx) {
  camera_ = new df::Pinhole(1241, 376, 718.856, 718.856, 607.1928, 185.2157);
  root_dir_= std::getenv("DATA_KITTI");
  io_ = new IO(root_dir_+"/00");
  cout << "Read " << io_->n_imgs() << " files\n";

  load_ref();
}

DepthAnalysis::~DepthAnalysis() {
  delete camera_;
  delete io_;
}

void DepthAnalysis::load_ref() {
  io_->read_set(ref_idx_, ref_ts_, ref_img_, T_w_ref_);  
  io_->read_vel(ref_idx_, &cloud_);
  cout << "Read " << cloud_.npts() << " velodyne points" << endl;
  auto img_ref_copy = ref_img_.clone();

  // detect features in ref image
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  vector<cv::KeyPoint> kps_ref;
  cv::Mat des_ref;
  orb->detectAndCompute(ref_img_, cv::noArray(), kps_ref, des_ref);
  std::for_each(kps_ref.begin(), kps_ref.end(), [&](cv::KeyPoint &kpt) {
    ref_kps_.emplace_back(kpt.pt);
  });
}

void DepthAnalysis::run_two_view(size_t cur_idx) {
  // load current image data
  cout << "Analysis between; ref = " << ref_idx_ << "\t cur = " << cur_idx << endl;

  cv::Mat img_cur;
  double ts_cur;
  Sophus::SE3 T_w_cur;
  io_->read_set(cur_idx, ts_cur, img_cur, T_w_cur);
  auto img_ref_copy = ref_img_.clone();
  auto img_cur_copy = img_cur.clone();
  img_ref_copy.convertTo(img_ref_copy, CV_32F);
  img_cur_copy.convertTo(img_cur_copy, CV_32F);

#ifdef DEBUG_YES
  cout << "#features = " << ref_kps_.size() << "\n";
  auto temp_img = ref_img_.clone();
  cv::cvtColor(temp_img, temp_img, CV_GRAY2BGR);
  std::for_each(ref_kps_.begin(), ref_kps_.end(), [&](cv::Point2f &kpt) {
    cv::circle(temp_img, kpt, 1, cv::Scalar(255, 0, 0));
  });
  cv::imshow("img_ref", temp_img);
  cv::waitKey(5000);
#endif

  // compute epiline
  auto T_cur_ref = T_w_cur.inverse() * T_w_ref_;
  auto R_cur_ref = T_cur_ref.rotation_matrix();
  auto t = T_cur_ref.translation();
  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1),
           t(2), 0, -t(0),
          -t(1), t(0), 0;
  auto E = R_cur_ref * t_hat;
  auto cam = static_cast<Pinhole*>(camera_);
  auto K = cam->K();
  auto F = K.transpose().inverse() * E * K.inverse();
  srand(time(0));
  size_t rand_idx = rand()%ref_kps_.size();
  auto kpt_ref = ref_kps_[rand_idx];
  auto kpt_cur = ref_kps_[rand_idx];
  float z_min = 1.0f;
  float z_max = 1000.0f;
  float dz = 1;
  Vector2d uv_ref(kpt_ref.x, kpt_ref.y);
  Vector3d f_vec_ref = camera_->cam2world(uv_ref);
  Vector3d px_homo(uv_ref.x(), uv_ref.y(), 1.0);
  Vector3d epiline = px_homo.transpose() * F;
  std::string filename = "/tmp/ref_"+to_string(ref_idx_)+"_cur_"+to_string(cur_idx)+".score";
  std::ofstream file(filename);
  file << ref_idx_ << " " << cur_idx << endl;
  file << uv_ref.x() << " " << uv_ref.y() << endl;
  while (z_min < z_max) {
    Vector3d pt_ref = f_vec_ref * z_min;
    Vector3d pt_cur = T_cur_ref * pt_ref;
    Vector2d uv_cur = camera_->world2cam(pt_cur);

    // calculate NCC score
    auto img_ref_norm = df::utils::normalize_image(
        img_ref_copy, kpt_ref.y-half_patch_size, kpt_ref.x-half_patch_size,
        patch_size, patch_size);
    auto img_cur_norm = df::utils::normalize_image(
        img_cur_copy, uv_cur.y()-half_patch_size, uv_cur.x()-half_patch_size,
        patch_size, patch_size);
    auto img_ref_ptr = img_ref_norm.ptr<float>();
    auto img_cur_ptr = img_cur_norm.ptr<float>();
    float ncc_score = df::utils::cross_correlation_single_patch(
        img_ref_ptr, img_cur_ptr, ref_img_.cols, img_cur.cols,
        kpt_ref.x-half_patch_size, kpt_ref.y-half_patch_size,
        uv_cur.x()-half_patch_size, uv_cur.y()-half_patch_size,
        patch_size, patch_size);
    cout  << "\t z_max = " << z_max
          << "\t z_cur = " << z_min
          << "\t xyz = " << pt_cur.transpose() 
          << "\t uv = " << uv_cur.transpose()
          << "\t ncc = " << ncc_score
          << endl;

    file << z_min << " " << ncc_score << endl;

    cv::Mat img_ref_n = ref_img_.clone();
    cv::Mat img_cur_n = img_cur.clone();
    cv::cvtColor(img_ref_n, img_ref_n, CV_GRAY2BGR);
    cv::cvtColor(img_cur_n, img_cur_n, CV_GRAY2BGR);

    cv::rectangle(img_ref_n, cv::Point2f(kpt_ref.x-4, kpt_ref.y-4), cv::Point2f(kpt_ref.x+4, kpt_ref.y+4), cv::Scalar(255, 0, 0));
    cv::rectangle(img_cur_n, cv::Point2f(uv_cur.x()-4, uv_cur.y()-4), cv::Point2f(uv_cur.x()+4, uv_cur.y()+4), cv::Scalar(0, 255, 0));
    draw_line(epiline, img_cur_n);

    cv::imshow("img_ref", img_ref_n);
    cv::imshow("img_cur", img_cur_n);
    cv::waitKey(0);
    z_min += dz;
  }
}

void DepthAnalysis::run_N_view() {

  size_t start_frame_idx = ref_idx;
  size_t end_frame_idx   = start_frame_idx + 100;

  while (start_frame_idx < end_frame_idx) {

  }
}

int main() {
  DepthAnalysis analyzer(ref_idx);
  analyzer.run_two_view(ref_idx+1);
}
