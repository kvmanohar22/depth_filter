#include <iostream>
#include <tuple>
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

static const int n_moves = 150; /// Number of moves along optical ray

static const int upper_thresh = 40;
static const int lower_thresh = 30;
static const float well_textured_thresh = 1e-2;

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
  bool run_two_view_single_patch(size_t cur_idx, size_t idx);
  void run_N_view();

protected:
  void load_ref();
  bool is_well_textured(Vector2i px);

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
  vector<tuple<Vector3d, cv::Point2f> > lidar_kps_;
  Sophus::SE3          T_cam0_vel_;
  vector<size_t>       cloud_order_;
  size_t               n_not_well_textured_;
};

DepthAnalysis::DepthAnalysis(size_t ref_idx)
  : ref_idx_(ref_idx),
    n_not_well_textured_(0) {
  camera_ = new df::Pinhole(1241, 376, 718.856, 718.856, 607.1928, 185.2157);
  root_dir_= std::getenv("DATA_KITTI");
  io_ = new IO(root_dir_+"/00");
  T_cam0_vel_ = io_->T_cam0_vel();
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

  // project all lidar points onto image
  std::for_each(cloud_.points_.begin(), cloud_.points_.end(), [&](Vector3f &xyz_lidar) {
    Vector3d xyz_ref = T_cam0_vel_ * xyz_lidar.cast<double>();
    if (xyz_ref.z() > 0) {
      Vector2d uv_ref = camera_->world2cam(xyz_ref);
      if (camera_->is_in_frame(uv_ref.cast<int>(), 8)) {
        if (is_well_textured(uv_ref.cast<int>())) {
          auto kpt = cv::Point2f(uv_ref.x(), uv_ref.y());
          auto obs = std::make_tuple(xyz_ref, kpt);
          lidar_kps_.push_back(obs);
        }
      }
    }
  });
  cloud_order_.resize(lidar_kps_.size());
  for(size_t i=0; i<lidar_kps_.size();++i)
    cloud_order_[i] = i;
  std::random_shuffle(cloud_order_.begin(), cloud_order_.end());
  cout << "Number of lidar points in ref = " << lidar_kps_.size() << endl;
  cout << "Number of lidar points not well textured = " << n_not_well_textured_ << endl;

  auto temp_img = ref_img_.clone();
  cv::cvtColor(temp_img, temp_img, CV_GRAY2BGR);
  std::for_each(lidar_kps_.begin(), lidar_kps_.end(), [&](tuple<Vector3d, cv::Point2f> &kpt) {
    cv::circle(temp_img, std::get<1>(kpt), 1, cv::Scalar(255, 0, 0));
  });
  cv::imshow("lidar points", temp_img);
  cv::waitKey(0);
}

bool DepthAnalysis::is_well_textured(Vector2i px) {
  auto img_temp = ref_img_.clone();
  img_temp.convertTo(img_temp, CV_32F);
  auto img_ptr = img_temp.ptr<float>();
  float mean = 0.0f;
  size_t max_r = px.y()+half_patch_size;
  size_t max_c = px.x()+half_patch_size;
  for (size_t r = px.y()-half_patch_size; r < max_r; ++r)
    for (size_t c = px.x()-half_patch_size; c < max_c; ++c)
      mean += *(img_ptr + r*img_temp.cols+c);
  mean /= (patch_size*patch_size);
  vector<float> residuals;
  for (size_t r = px.y()-half_patch_size; r < max_r; ++r) {
    for (size_t c = px.x()-half_patch_size; c < max_c; ++c) {
      float diff = *(img_ptr + r*img_temp.cols+c) - mean;
      auto diff2 = diff*diff;
      residuals.push_back(diff2);
    }
  }
  if (sqrt(std::accumulate(residuals.begin(), residuals.end(), 0.0f)) < well_textured_thresh) {
    ++n_not_well_textured_;
    return false;
  }
  return true;
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

  // Epipolar lines
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

  // Go through each lidar point and analyze stats
  for (size_t i = 0; i < lidar_kps_.size(); ++i) {
    cout << "Processing " << i << " / " << lidar_kps_.size() << " points\n";
    auto idx = cloud_order_[i];
    auto ref_pt = std::get<0>(lidar_kps_[idx]);
    auto ref_uv = std::get<1>(lidar_kps_[idx]);

    if (ref_pt.z() < upper_thresh)
      continue;

    // if (ref_pt.z() > lower_thresh)
    //   continue;

    float z_min = 0.2 * ref_pt.z();
    float z_max = 1.4 * ref_pt.z();
    float dz = 0.2f;

    Vector2d uv_ref(ref_uv.x, ref_uv.y);
    Vector3d f_vec_ref = camera_->cam2world(uv_ref);
    Vector3d px_homo(uv_ref.x(), uv_ref.y(), 1.0);
    Vector3d epiline = px_homo.transpose() * F;

    // TODO: change the file name dynamically
    std::string filename = "/tmp/df_upper_stats/ref_"+
      to_string(ref_idx_)+"_cur_"+to_string(cur_idx)+"_idx_"+to_string(i)+".score";
    std::ofstream file(filename);
    file << ref_idx_ << " " << cur_idx << endl;
    file << ref_uv.x << " " << ref_uv.y << endl;
    file << ref_pt.z() << " " << i << endl;

    auto z_vec = df::utils::linspace(z_min, z_max, n_moves);

    for (const auto &itr: z_vec) {
      Vector3d pt_ref = f_vec_ref * itr;
      Vector3d pt_cur = T_cur_ref * pt_ref;
      Vector2d uv_cur = camera_->world2cam(pt_cur);

      if (!camera_->is_in_frame(uv_cur.cast<int>())) {
        continue;
      }

      // calculate NCC score
      auto img_ref_norm = df::utils::normalize_image(
          img_ref_copy, ref_uv.y-half_patch_size, ref_uv.x-half_patch_size,
          patch_size, patch_size);
      auto img_cur_norm = df::utils::normalize_image(
          img_cur_copy, uv_cur.y()-half_patch_size, uv_cur.x()-half_patch_size,
          patch_size, patch_size);
      auto img_ref_ptr = img_ref_norm.ptr<float>();
      auto img_cur_ptr = img_cur_norm.ptr<float>();
      float ncc_score = df::utils::cross_correlation_single_patch(
          img_ref_ptr, img_cur_ptr, ref_img_.cols, img_cur.cols,
          ref_uv.x-half_patch_size, ref_uv.y-half_patch_size,
          uv_cur.x()-half_patch_size, uv_cur.y()-half_patch_size,
          patch_size, patch_size);
      cout  << "z_min = " << z_min
            << "\t z_true = " << ref_pt.z()
            << "\t z_cur = " << itr
            << "\t z_max = " << z_max
            << "\t uv = " << uv_cur.transpose()
            << "\t ncc = " << ncc_score
            << endl;

      file << itr << " " << ncc_score << endl;

      cv::Mat img_ref_n = ref_img_.clone();
      cv::Mat img_cur_n = img_cur.clone();
      cv::cvtColor(img_ref_n, img_ref_n, CV_GRAY2BGR);
      cv::cvtColor(img_cur_n, img_cur_n, CV_GRAY2BGR);

      cv::rectangle(img_ref_n,
          cv::Point2f(ref_uv.x-half_patch_size, ref_uv.y-half_patch_size),
          cv::Point2f(ref_uv.x+half_patch_size, ref_uv.y+half_patch_size), cv::Scalar(255, 0, 0));
      cv::rectangle(img_cur_n,
          cv::Point2f(uv_cur.x()-half_patch_size, uv_cur.y()-half_patch_size),
          cv::Point2f(uv_cur.x()+half_patch_size, uv_cur.y()+half_patch_size), cv::Scalar(0, 255, 0));
      draw_line(epiline, img_cur_n);

      cv::imshow("img_ref", img_ref_n);
      cv::imshow("img_cur", img_cur_n);
      cv::waitKey(10);
    }
  }
}

bool DepthAnalysis::run_two_view_single_patch(size_t cur_idx, size_t idx) {
  // load current image data
  cv::Mat img_cur;
  double ts_cur;
  Sophus::SE3 T_w_cur;
  io_->read_set(cur_idx, ts_cur, img_cur, T_w_cur);
  auto img_ref_copy = ref_img_.clone();
  auto img_cur_copy = img_cur.clone();
  img_ref_copy.convertTo(img_ref_copy, CV_32F);
  img_cur_copy.convertTo(img_cur_copy, CV_32F);

  // Epipolar lines
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

  // Go through each lidar point and analyze stats
  auto pt_idx = cloud_order_[idx];
  auto ref_pt = std::get<0>(lidar_kps_[pt_idx]);
  auto ref_uv = std::get<1>(lidar_kps_[pt_idx]);

  // if (ref_pt.z() < upper_thresh)
  //   return false;

  if (ref_pt.z() > lower_thresh) {
    cout << "WARNING: Too far away from camera center!\n";
    return false;
  }

#ifdef DEBUG_YES
  cout << "\t ref = " << ref_idx_ << "\t cur = " << cur_idx << endl;
#endif

  float z_min = 0.05 * ref_pt.z();
  float z_max = 1.4 * ref_pt.z();
  float dz = 0.2f;

  Vector2d uv_ref(ref_uv.x, ref_uv.y);
  Vector3d f_vec_ref = camera_->cam2world(uv_ref);
  Vector3d px_homo(uv_ref.x(), uv_ref.y(), 1.0);
  px_homo.normalize();
  Vector3d epiline = px_homo.transpose() * F;

  std::string df_dir = std::getenv("PROJECT_DF");
  std::string new_dir = df_dir+"/analysis/logs/df_lower_stats_nview/idx_"+
    to_string(idx)+"_point_"+to_string(cloud_order_[idx]);
  std::string filename = new_dir+"/scores"+"/ref_"+to_string(ref_idx_)+"_cur_"+to_string(cur_idx)+".score";
  if (!boost::filesystem::exists(new_dir)) {
    boost::filesystem::create_directory(new_dir);
    boost::filesystem::create_directory(new_dir+"/scores");
  }
  std::ofstream file(filename);
  file << ref_idx_ << " " << cur_idx << endl;
  file << ref_uv.x << " " << ref_uv.y << endl;
  file << ref_pt.z() << " " << idx << endl;

  auto z_vec = df::utils::linspace(z_min, z_max, n_moves);

  for (size_t ii = 0; ii < z_vec.size(); ++ii) {
    Vector3d pt_ref = f_vec_ref * z_vec[ii];
    Vector3d pt_cur = T_cur_ref * pt_ref;
    Vector2d uv_cur = camera_->world2cam(pt_cur);

    if (!camera_->is_in_frame(uv_cur.cast<int>())) {
      cout << "WARNING: Not in frame...skipping\n";
      continue;
    }

    // calculate NCC score
    auto img_ref_norm = df::utils::normalize_image(
        img_ref_copy, ref_uv.y-half_patch_size, ref_uv.x-half_patch_size,
        patch_size, patch_size);
    auto img_cur_norm = df::utils::normalize_image(
        img_cur_copy, uv_cur.y()-half_patch_size, uv_cur.x()-half_patch_size,
        patch_size, patch_size);
    auto img_ref_ptr = img_ref_norm.ptr<float>();
    auto img_cur_ptr = img_cur_norm.ptr<float>();
    float ncc_score = df::utils::cross_correlation_single_patch(
        img_ref_ptr, img_cur_ptr, ref_img_.cols, img_cur.cols,
        ref_uv.x-half_patch_size, ref_uv.y-half_patch_size,
        uv_cur.x()-half_patch_size, uv_cur.y()-half_patch_size,
        patch_size, patch_size);

    file << z_vec[ii] << " " << ncc_score << endl;

#ifdef DEBUG_YES
    cout  << "z_min = " << z_min
          << "\t z_true = " << ref_pt.z()
          << "\t z_cur = " << z_vec[ii]
          << "\t z_max = " << z_max
          << "\t uv = " << uv_cur.transpose()
          << "\t ncc = " << ncc_score
          << endl;

    cv::Mat img_ref_n = ref_img_.clone();
    cv::Mat img_cur_n = img_cur.clone();
    cv::cvtColor(img_ref_n, img_ref_n, CV_GRAY2BGR);
    cv::cvtColor(img_cur_n, img_cur_n, CV_GRAY2BGR);

    cv::rectangle(img_ref_n,
        cv::Point2f(ref_uv.x-half_patch_size, ref_uv.y-half_patch_size),
        cv::Point2f(ref_uv.x+half_patch_size, ref_uv.y+half_patch_size), cv::Scalar(255, 0, 0));
    cv::rectangle(img_cur_n,
        cv::Point2f(uv_cur.x()-half_patch_size, uv_cur.y()-half_patch_size),
        cv::Point2f(uv_cur.x()+half_patch_size, uv_cur.y()+half_patch_size), cv::Scalar(0, 255, 0));
    draw_line(epiline, img_cur_n);

    cv::imshow("img_ref", img_ref_n);
    cv::imshow("img_cur", img_cur_n);
    cv::waitKey(1);
#endif
  }
  return true;
}

void DepthAnalysis::run_N_view() {

  size_t current_frame_idx = ref_idx_+1;
  size_t end_frame_idx   = current_frame_idx + 20;

  for (size_t i = 0; i < lidar_kps_.size(); ++i) {
    cout << "Processing " << i << " / " << lidar_kps_.size() << endl;
    size_t current_frame_idx = ref_idx_+1;
    while (current_frame_idx < end_frame_idx) {
      if(!run_two_view_single_patch(current_frame_idx, i))
        break;
      ++current_frame_idx;
    }
  }
}

int main() {
  DepthAnalysis analyzer(ref_idx);
  // analyzer.run_two_view(ref_idx+1);
  analyzer.run_N_view();
}
