#include <iostream>
#include <tuple>
#include <cstdio>

#include "depth_filter/global.hpp"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/cloud.h"
#include "depth_filter/depth_filter.h"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"

using namespace io;
using namespace std;
using namespace df;

static const int IMG_H = 376;
static const int IMG_W = 1241;
static const int patch_size = 4;
static const int half_patch_size = patch_size / 2;

static const int ref_idx = 1030;

static const int upper_thresh = 40;
static const int lower_thresh = 20;
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

class DepthFilterTest {
public:
  DepthFilterTest(size_t ref_idx);
  ~DepthFilterTest();
  void run();

protected:
  void load_ref();
  bool is_well_textured(Vector2i px);

private:
  string          root_dir_;
  AbstractCamera* camera_;
  IO*             io_;

  /// Reference image stuff
  size_t               ref_idx_;
  double               ref_ts_;
  cv::Mat              ref_img_;
  Sophus::SE3          T_w_ref_;
  PointCloud           cloud_;
  DepthFilter*         depth_filter_;
  vector<cv::Point2f>  ref_kps_;
  vector<tuple<Vector3d, cv::Point2f> > lidar_kps_;
  Sophus::SE3          T_cam0_vel_;
  vector<size_t>       cloud_order_;
  size_t               n_not_well_textured_;

  /// frames
  FramePtr             frame_ref_;
  FramePtr             frame_cur_;
};

DepthFilterTest::DepthFilterTest(size_t ref_idx)
  : ref_idx_(ref_idx),
    n_not_well_textured_(0),
    depth_filter_(nullptr) {
  camera_ = new df::Pinhole(1241, 376, 718.856, 718.856, 607.1928, 185.2157);
  root_dir_= std::getenv("DATA_KITTI");
  io_ = new Kitti(root_dir_+"/00");
  T_cam0_vel_ = dynamic_cast<io::Kitti*>(io_)->T_cam0_vel();
  DLOG(INFO) << "Read " << io_->n_imgs() << " files";

  load_ref();
}

DepthFilterTest::~DepthFilterTest() {
  delete camera_;
}

bool DepthFilterTest::is_well_textured(Vector2i px) {
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

void DepthFilterTest::load_ref() {
  io_->read_set(ref_idx_, ref_ts_, ref_img_, T_w_ref_);  
  dynamic_cast<io::Kitti*>(io_)->read_vel(ref_idx_, &cloud_);
  DLOG(INFO) << "Read " << cloud_.npts() << " velodyne points";
  auto img_ref_copy = ref_img_.clone();

  // project all lidar points onto image
  std::for_each(cloud_.points_.begin(), cloud_.points_.end(), [&](Vector3f &xyz_lidar) {
    Vector3d xyz_ref = T_cam0_vel_ * xyz_lidar.cast<double>();
    // if (xyz_ref.z() > 0 && xyz_ref.z() > upper_thresh) {
    if (xyz_ref.z() > 0 && xyz_ref.z() < lower_thresh) {
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
  DLOG(INFO) << "Number of lidar points in ref = " << lidar_kps_.size();
  DLOG(INFO) << "Number of lidar points not well textured = " << n_not_well_textured_;

  auto temp_img = ref_img_.clone();
  cv::cvtColor(temp_img, temp_img, CV_GRAY2BGR);
  std::for_each(lidar_kps_.begin(), lidar_kps_.end(), [&](tuple<Vector3d, cv::Point2f> &kpt) {
    cv::circle(temp_img, std::get<1>(kpt), 1, cv::Scalar(255, 0, 0));
  });
  cv::imshow("lidar points", temp_img);
  cv::waitKey(0);
}

void DepthFilterTest::run() {
  float depth_mean = 10.0f, depth_min = 1.0f, depth_max = 50.0f;

  for (size_t i=ref_idx_; i < ref_idx_+50; ++i) {
    // Load data
    double frame_ts; cv::Mat frame_img; Sophus::SE3 T_w_f;
    io_->read_set(i, frame_ts, frame_img, T_w_f);

    // initialize the depth filter
    if (i == ref_idx_) {
      frame_ref_.reset(new Frame(i, frame_img, camera_, frame_ts));
      frame_ref_->set_pose(T_w_f);
      depth_filter_ = new DepthFilter();
      depth_filter_->add_keyframe(frame_ref_);
      list<Seed>& seeds = depth_filter_->get_mutable_seeds();
      for (size_t i=0; i<cloud_order_.size(); ++i) {
        std::tuple<Vector3d, cv::Point2f>& kpt = lidar_kps_[cloud_order_[i]];
        auto pt = std::get<1>(kpt);
        Vector2d px(pt.x, pt.y);
        Corner* new_corner = new Corner(px, frame_ref_);
        seeds.emplace_back(Seed(depth_mean, depth_min, depth_max, new_corner));
      }
      DLOG(INFO) << "Number of initialized seeds = " << depth_filter_->n_seeds();
      continue;
    }

    // Now update the initialized seeds
    frame_cur_.reset(new Frame(i, frame_img, camera_, frame_ts));
    frame_cur_->set_pose(T_w_f);
    depth_filter_->add_frame(frame_cur_);
  }  
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]); 

  DepthFilterTest tester(ref_idx);
  tester.run();
}
