#include "depth_filter/global.h"
#include "depth_filter/cameras/abstract.h"
#include "depth_filter/cameras/pinhole.h"
#include "depth_filter/io.h"
#include "depth_filter/utils.h"
#include "depth_filter/feature_detector.h"
#include "depth_filter/depth_filter.h"
#include "depth_filter/similarity.h"

using namespace std;
using namespace df;

namespace {

static constexpr size_t half_patch_size = 4;
static constexpr size_t patch_size = 2*half_patch_size;

class SearchSimilarityTest {
public:
  explicit SearchSimilarityTest(size_t ref_idx, string logdir);
  ~SearchSimilarityTest();
  void run(Corner* corner, size_t cidx);
  void load_ref();
  void run_two_view(FramePtr& frame_cur, Corner* corner, size_t cidx, size_t cur_idx);
  void run_N_view();

private:
  size_t              ref_idx_;
  AbstractCamera*     camera_;
  io::IO*             io_;
  cv::Mat             depth_img_ref_;
  FramePtr            frame_ref_;
  list<Corner*>       corners_;
  string              logdir_root_;
  ::utils::similarity::ZMNCC<uint8_t, half_patch_size>* zmncc_;
  uint8_t             ref_patch_[patch_size*patch_size] __attribute__ ((aligned(16)));
  bool                verbose_;
};

SearchSimilarityTest::SearchSimilarityTest(size_t ref_idx, string logdir)
  : ref_idx_(ref_idx), logdir_root_(logdir), verbose_(true)
{
  camera_ = new df::Pinhole(752, 480, 315.5, 315.5, 376.0, 240.0);
  io_ = new io::RPGSyntheticDownward(std::getenv("DATA_RPG_SYNTHETIC_DOWNWARD"));
  DLOG(INFO) << "Read " << io_->n_imgs() << " files";
  load_ref();
  if (!boost::filesystem::exists(logdir)) {
    boost::filesystem::create_directory(logdir);
  }
  zmncc_ = new ::utils::similarity::ZMNCC<uint8_t, half_patch_size>();
}

SearchSimilarityTest::~SearchSimilarityTest() {
  delete camera_;
  delete io_;
  delete zmncc_;
}

void SearchSimilarityTest::load_ref() {
  double frame_ts; cv::Mat frame_img; Sophus::SE3 T_w_f;
  io_->read_set(ref_idx_, frame_ts, frame_img, T_w_f);
  frame_ref_.reset(new Frame(ref_idx_, frame_img, camera_, frame_ts));
  frame_ref_->set_pose(T_w_f);

  FastDetector::detect(frame_ref_, corners_);
  // In this dataset, depth is stored along optical axis
  dynamic_cast<io::RPGSyntheticDownward*>(io_)->read_gtdepth(
    ref_idx_, camera_, depth_img_ref_, df::utils::DepthType::OPTICAL_AXIS);

  float* img_ptr = depth_img_ref_.ptr<float>();
  for(auto itr=corners_.begin(); itr!=corners_.end();) {
    Vector2i px = (*itr)->px_.cast<int>();
    if(!camera_->is_in_frame(px, 128)) {
      itr=corners_.erase(itr);
      continue;
    }
    float scale = depth_img_ref_.at<float>((*itr)->px_.y(), (*itr)->px_.x());
    Vector3d p = (*itr)->f_*scale;
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

void SearchSimilarityTest::run(Corner* corner, size_t cidx) {
  FramePtr frame_cur;
  for (size_t i=ref_idx_+1; i < ref_idx_+20; ++i) {
    // Load data
    double frame_ts; cv::Mat frame_img; Sophus::SE3 T_w_f;
    io_->read_set(i, frame_ts, frame_img, T_w_f);

    frame_cur.reset(new Frame(i, frame_img, camera_, frame_ts));
    frame_cur->set_pose(T_w_f);

    // Run the analysis
    run_two_view(frame_cur, corner, cidx, i);
  }
}

void SearchSimilarityTest::run_N_view() {
  size_t cidx=0;
  std::for_each(corners_.begin(), corners_.end(), [&](Corner* new_corner) {
    LOG(INFO) << cidx << " / " << corners_.size();
    string corner_dirname = logdir_root_+"/corner_idx_"+to_string(cidx);
    boost::filesystem::create_directory(corner_dirname.c_str());

    // set the reference patch
    DepthFilter::create_ref_patch(ref_patch_, new_corner->px_, frame_ref_->img_, half_patch_size);
    zmncc_->set_ref_patch(ref_patch_);

    // run the analysis
    this->run(new_corner, cidx);
    ++cidx;
  });
}

void SearchSimilarityTest::run_two_view(FramePtr& frame_cur, Corner* corner, size_t cidx, size_t cur_idx) {
  float focal_len = dynamic_cast<df::Pinhole*>(camera_)->fx();
  const float one_px_angle = 2.0*atan(1.0/(2.0*focal_len));
  
  const auto T_cur_ref = frame_cur->T_f_w_*corner->frame_->T_f_w_.inverse();
  const auto px_ref = corner->px_;
  const auto rp = corner->xyz_->xyz();
  const auto t  = T_cur_ref.inverse().translation();
  const auto f_unit = corner->f_;
  float tau = fabs(DepthFilter::compute_tau(rp, t, f_unit, one_px_angle));

  float depth_along_ray = rp.z()/f_unit.z();
  float d_min = depth_along_ray - 30*tau;
  float d_max = depth_along_ray + 30*tau;

  // points on the unit sphere
  Vector2d pt_min = df::utils::project2d(T_cur_ref*(corner->f_*d_min));
  Vector2d pt_max = df::utils::project2d(T_cur_ref*(corner->f_*d_max));

  // pixels corresponding to extrema
  Vector2d uv_min = frame_ref_->cam_->world2cam(T_cur_ref*(corner->f_*d_min));
  Vector2d uv_max = frame_ref_->cam_->world2cam(T_cur_ref*(corner->f_*d_max));
  Vector2d uv_best = frame_ref_->cam_->world2cam(T_cur_ref*rp);

#ifdef DEBUG_YES
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
#endif

  // go along the line and compute similarity measure
  float linelen = (uv_max - uv_min).norm();
  size_t n_steps = linelen / 0.7;
  Vector2d linedir = uv_max - uv_min;
  Vector2d step = linedir / n_steps;
  if (n_steps > 1000) {
    LOG(WARNING) << "n_steps = " << n_steps << " is large...! Skipping";
    return;
  }
  Vector2d uv = pt_min+step;
  Vector2i uv_last(0, 0);
  std::ofstream ofile(
    logdir_root_+"/corner_idx_"+to_string(cidx)+
    "/ref_"+to_string(ref_idx_)+"_cur_"+to_string(cur_idx)+".txt");
  for(size_t i=0;i<n_steps; ++i, uv+=step) {
    const Vector2d uv_cur(frame_cur->cam_->world2cam(uv));
    Vector2i uv_int(uv_cur.x()+0.5, uv_cur.y()+0.5);
    if (uv_last == uv_int) {
      continue;
    }
    uv_last = uv_int;
    if (!frame_cur->cam_->is_in_frame(uv_int, patch_size)) {
      continue;
    }

    uint8_t* cur_patch_ptr = frame_cur->img_.data
                             + (uv_int[1]-half_patch_size)*frame_cur->img_.cols
                             + (uv_int[0]-half_patch_size);
    double similarity = zmncc_->similarity(cur_patch_ptr, frame_cur->img_.cols);
    float depth_cur;
    if(DepthFilter::triangulate(T_cur_ref, corner->f_, df::utils::unproject2d(uv).normalized(), depth_cur))
      ofile << depth_cur << " " << similarity << std::endl;
    else
      ofile << -1 << " " << similarity << " " << std::endl;

    // visualize
#ifdef DEBUG_YES
  imgr = frame_ref_->img_.clone();
  imgc = frame_cur->img_.clone();
  cv::cvtColor(imgr, imgr, CV_GRAY2BGR);
  cv::cvtColor(imgc, imgc, CV_GRAY2BGR);
  cv::circle(imgr, cv::Point2f(px_ref.x(), px_ref.y()), 1, cv::Scalar(255, 0, 0), 1); 
  cv::circle(imgc, cv::Point2f(uv_best.x(), uv_best.y()), 1, cv::Scalar(0, 0, 255), 1); 
  cv::circle(imgc, cv::Point2f(uv_cur.cast<float>().x(), uv_cur.cast<float>().y()), 1, cv::Scalar(0, 0, 255), 1); 
  cv::line(imgc, cv::Point2f(uv_min.x(), uv_min.y()), cv::Point2f(uv_max.x(), uv_max.y()), cv::Scalar(255, 0, 0), 1); 
  cv::imshow("ref", imgr); 
  cv::imshow("cur", imgc); 
  cv::waitKey(0);
#endif
  }

}

}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]); 

  size_t ref_idx = 0;
  string logdir = std::getenv("PROJECT_DF");
  logdir = logdir+"/analysis/downward/log3";
  SearchSimilarityTest tester(ref_idx, logdir);
  tester.run_N_view();
}
