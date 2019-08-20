#include <iostream>

#include "depth_filter/global.hpp"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"

using namespace io;
using namespace std;
using namespace df;

static const int IMG_H = 376;
static const int IMG_W = 1241;

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
  io_ = new IO(root_dir_+"/02");
  cout << "Read " << io_->n_imgs() << " files\n";
}

DepthAnalysis::~DepthAnalysis() {
  delete camera_;
  delete io_;
}

void DepthAnalysis::run_two_view() {
  // load images and GT poses
  cv::Mat img_ref, img_cur;
  double ts_ref, ts_cur;
  Sophus::SE3 T_w_ref, T_w_cur;
  io_->read_set(2, ts_ref, img_ref, T_w_ref);  
  io_->read_set(9, ts_cur, img_cur, T_w_cur);

#ifdef DEBUG_YES
  cv::imshow("ref", img_ref);
  cv::imshow("cur", img_cur);
  cv::waitKey(3000);
  cv::destroyWindow("ref");
  cv::destroyWindow("cur");
#endif

  // detect features in ref image
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  vector<cv::KeyPoint> kps_ref;
  cv::Mat des_ref;
  orb->detectAndCompute(img_ref, cv::noArray(), kps_ref, des_ref); 

#ifdef DEBUG_YES
  cout << "#features = " << kps_ref.size() << "\n";
  auto temp_img = img_ref.clone();
  cv::cvtColor(temp_img, temp_img, CV_GRAY2BGR);
  std::for_each(kps_ref.begin(), kps_ref.end(), [&](cv::KeyPoint &kpt) {
    cv::circle(temp_img, kpt.pt, 1, cv::Scalar(255, 0, 0));
  });
  cv::imshow("img_ref", temp_img);
  cv::waitKey(0);
#endif

  auto T_cur_ref = T_w_cur.inverse() * T_w_ref;
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
  size_t rand_idx = rand()%kps_ref.size();
  auto kpt = kps_ref[rand_idx].pt;
  float z_min = 1.0f;
  float z_max = 1000.0f;
  float dz = 1;
  Vector2d px(kpt.x, kpt.y);
  Vector3d f_vec_ref = camera_->cam2world(px);
  Vector3d px_homo(px.x(), px.y(), 1.0);
  Vector3d epiline = px_homo.transpose() * F;
  while (z_min < z_max) {
    Vector3d pt_ref = f_vec_ref * z_min;
    Vector3d pt_cur = T_cur_ref * pt_ref;
    Vector2d uv_cur = camera_->world2cam(pt_cur);
    cout  << "\t z_max = " << z_max
          << "\t z_cur = " << z_min
          << "\t xyz = " << pt_cur.transpose() 
          << "\t uv = " << uv_cur.transpose()
          << endl;
    cv::Mat img_ref_n = img_ref.clone();
    cv::Mat img_cur_n = img_cur.clone();
    cv::cvtColor(img_ref_n, img_ref_n, CV_GRAY2BGR);
    cv::cvtColor(img_cur_n, img_cur_n, CV_GRAY2BGR);

    cv::rectangle(img_ref_n, cv::Point2f(kpt.x-4, kpt.y-4), cv::Point2f(kpt.x+4, kpt.y+4), cv::Scalar(255, 0, 0));
    cv::rectangle(img_cur_n, cv::Point2f(uv_cur.x()-4, uv_cur.y()-4), cv::Point2f(uv_cur.x()+4, uv_cur.y()+4), cv::Scalar(0, 255, 0));
    draw_line(epiline, img_cur_n);

    cv::imshow("img_ref", img_ref_n);
    cv::imshow("img_cur", img_cur_n);
    cv::waitKey(0);
    z_min += dz;
  }
}

int main() {
  DepthAnalysis analyzer;
  analyzer.run_two_view();
}
