#include "depth_filter/global.h"
#include "depth_filter/cameras/abstract.h"
#include "depth_filter/cameras/pinhole.h"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/feature_detector.h"
#include "depth_filter/depth_filter.h"

using namespace std;
using namespace io;
using namespace df;

namespace {

class SearchSimilarityTest {
public:
  explicit SearchSimilarityTest(size_t ref_idx);
  ~SearchSimilarityTest();
  void run();
  void load_ref();
  void run_two_view(FramePtr& frame_cur, Corner* corner);

private:
  size_t              ref_idx_;
  AbstractCamera*     camera_;
  IO*                 io_;
  cv::Mat             depth_img_ref_;
  FramePtr            frame_ref_;
  list<Corner*>       corners_;
};

SearchSimilarityTest::SearchSimilarityTest(size_t ref_idx)
  : ref_idx_(ref_idx)
{
  camera_ = new df::Pinhole(752, 480, 315.5, 315.5, 376.0, 240.0);
  io_ = new RPGSyntheticDownward(std::getenv("DATA_RPG_SYNTHETIC_DOWNWARD"));
  DLOG(INFO) << "Read " << io_->n_imgs() << " files";
  load_ref();
}

SearchSimilarityTest::~SearchSimilarityTest() {
  delete camera_;
  delete io_;
}

void SearchSimilarityTest::load_ref() {
  double frame_ts; cv::Mat frame_img; Sophus::SE3 T_w_f;
  io_->read_set(ref_idx_, frame_ts, frame_img, T_w_f);
  frame_ref_.reset(new Frame(ref_idx_, frame_img, camera_, frame_ts));
  frame_ref_->set_pose(T_w_f);

  FastDetector::detect(frame_ref_, corners_);
  dynamic_cast<io::RPGSyntheticDownward*>(io_)->read_gtdepth(
    ref_idx_, camera_, depth_img_ref_);

  auto img_ptr = depth_img_ref_.ptr<float>();
  for(auto itr=corners_.begin(); itr!=corners_.end();) {
    Vector2i px = (*itr)->px_.cast<int>();
    if(!camera_->is_in_frame(px, 128)) {
      itr=corners_.erase(itr);
      continue;
    }
    float depth = *(img_ptr+px.y()*depth_img_ref_.cols+px.x()); // this depth is along optical axis
    float depth_along_ray = depth / (float)(*itr)->f_.z(); // this depth is along optical ray
    Vector3d p = (*itr)->f_*depth_along_ray;
    auto point = new Point(p, *itr);
    (*itr)->xyz_ = point;
    ++itr;
  }

  auto img = frame_img.clone();
  cv::cvtColor(img, img, CV_GRAY2BGR);
  std::for_each(corners_.begin(), corners_.end(), [&](Corner *corner) {
    Vector2d px = corner->px_;
    cv::circle(img, cv::Point2f(px.x(), px.y()), 1, cv::Scalar(0, 255, 0), 1);
  });
  cv::imshow("ref image with corners", img);
  cv::waitKey(0);
}

void SearchSimilarityTest::run() {
  FramePtr frame_cur;
  for (size_t i=ref_idx_+1; i < ref_idx_+50; ++i) {
    // Load data
    double frame_ts; cv::Mat frame_img; Sophus::SE3 T_w_f;
    io_->read_set(i, frame_ts, frame_img, T_w_f);

    frame_cur.reset(new Frame(i, frame_img, camera_, frame_ts));
    frame_cur->set_pose(T_w_f);

    // Run the analysis
    run_two_view(frame_cur, corners_.back());
  }
}

void SearchSimilarityTest::run_two_view(FramePtr& frame_cur, Corner* corner) {
  float focal_len = dynamic_cast<df::Pinhole*>(camera_)->fx();
  const float one_px_angle = 2.0*atan(1.0/(2.0*focal_len));
  
  const auto T_cur_ref = frame_cur->T_f_w_*corner->frame_->T_f_w_.inverse();
  const auto px_ref = corner->px_;
  const auto rp = corner->xyz_->xyz();
  const auto t  = T_cur_ref.inverse().translation();
  const auto f_unit = corner->f_;
  float tau = fabs(DepthFilter::compute_tau(rp, t, f_unit, one_px_angle));

  float depth_along_ray = rp.z()/f_unit.z();
  float d_min = depth_along_ray - 10*tau;
  float d_max = depth_along_ray + 10*tau;
  Vector2d uv_min = frame_ref_->cam_->world2cam(T_cur_ref*(corner->f_*d_min));
  Vector2d uv_max = frame_ref_->cam_->world2cam(T_cur_ref*(corner->f_*d_max));
  Vector2d uv_best = frame_ref_->cam_->world2cam(T_cur_ref*rp);

  cv::Mat imgr = frame_ref_->img_.clone();
  cv::Mat imgc = frame_cur->img_.clone();
  cv::cvtColor(imgr, imgr, CV_GRAY2BGR);
  cv::cvtColor(imgc, imgc, CV_GRAY2BGR);
  cv::circle(imgr, cv::Point2f(px_ref.x(), px_ref.y()), 1, cv::Scalar(255, 0, 0), 1); 
  cv::circle(imgc, cv::Point2f(uv_best.x(), uv_best.y()), 1, cv::Scalar(0, 0, 255), 1); 
  cv::line(imgc, cv::Point2f(uv_min.x(), uv_min.y()), cv::Point2f(uv_max.x(), uv_max.y()), cv::Scalar(255, 0, 0), 1); 
  cv::imshow("ref", imgr); 
  cv::imshow("cur", imgc); 
  cv::waitKey(0);
}

}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]); 

  size_t ref_idx = 0;
  SearchSimilarityTest tester(ref_idx);
  tester.run();
}
