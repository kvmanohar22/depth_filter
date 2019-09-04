#include "depth_filter/global.hpp"
#include "depth_filter/utils.h"
#include "depth_filter/cameras/abstract.hpp"
#include "depth_filter/cameras/pinhole.hpp"
#include "depth_filter/io.h"
#include "depth_filter/depth_filter.h"

using namespace std;
using namespace io;
using namespace df;

namespace {

class DepthFilterTest {
public:
  DepthFilterTest(size_t ref_idx);
  ~DepthFilterTest() { delete camera_; }
  void load_ref();
  void run();

private:
  size_t              ref_idx_;
  string              root_dir_;
  vector<Sophus::SE3> poses_;
  vector<double>      ts_;
  AbstractCamera*     camera_;
  IO*                 io_;
  DepthFilter*        depth_filter_;
  cv::Mat             depth_img_ref_; 
  vector<tuple<float, cv::Point2f> > gt_kps_;

  /// frames
  FramePtr             frame_ref_;
  FramePtr             frame_cur_;
};

DepthFilterTest::DepthFilterTest(size_t ref_idx)
  : ref_idx_(ref_idx)
{
  camera_ = new df::Pinhole(752, 480, 315.5, 315.5, 376.0, 240.0);
  root_dir_= std::getenv("DATA_RPG_SYNTHETIC_DOWNWARD");
  io_ = new RPGSyntheticDownward(root_dir_);
  DLOG(INFO) << "Read " << io_->n_imgs() << " files";
}

void DepthFilterTest::load_ref() {
  dynamic_cast<io::RPGSyntheticDownward*>(io_)->read_gtdepth(ref_idx_, camera_, depth_img_ref_);

  float* img_ptr = depth_img_ref_.ptr<float>();
  float depth;
  gt_kps_.reserve(depth_img_ref_.rows*depth_img_ref_.cols);
  for(int y=8; y<camera_->height()-8; ++y) {
    for(int x=8; x<camera_->width()-8; ++x) {
      depth = *(img_ptr+y*depth_img_ref_.cols+x);
      gt_kps_.emplace_back(make_tuple(depth, cv::Point2f(x, y)));
    }
  }
  srand(time(0));
  std::random_shuffle(gt_kps_.begin(), gt_kps_.end());
  auto itr=gt_kps_.begin()+0.5*gt_kps_.size();
  std::nth_element(
      gt_kps_.begin(), itr, gt_kps_.end(),
      [&](tuple<float, cv::Point2f> a1, tuple<float, cv::Point2f> a2) {
    return std::get<0>(a1) < std::get<0>(a2);
  });
  LOG(INFO) << "50-percentile depth = " << std::get<0>(*itr);
}

void DepthFilterTest::run() {
  float depth_mean = 2.0f, depth_min = 1.0f, depth_max = 5.0f;

  for (size_t i=ref_idx_; i < ref_idx_+50; ++i) {
    // Load data
    double frame_ts; cv::Mat frame_img; Sophus::SE3 T_w_f;
    io_->read_set(i, frame_ts, frame_img, T_w_f);

    // initialize the depth filter
    if (i == ref_idx_) {
      frame_ref_.reset(new Frame(i, frame_img, camera_, frame_ts));
      frame_ref_->set_pose(T_w_f);
      depth_filter_ = new DepthFilter(camera_);
      depth_filter_->options_.verbose_ = true;
      depth_filter_->options_.seeds_init_type_ = SeedsInitType::FAST_DETECTOR;
      depth_filter_->add_keyframe(frame_ref_);
      // load_ref();
      // list<Seed>& seeds = depth_filter_->get_mutable_seeds();
      // for (size_t i=0; i<gt_kps_.size(); ++i) {
      //   std::tuple<float, cv::Point2f>& kpt = gt_kps_[i];
      //   auto pt = std::get<1>(kpt);
      //   Vector2d px(pt.x, pt.y);
      //   Corner* new_corner = new Corner(px, frame_ref_);
      //   seeds.emplace_back(Seed(depth_mean, depth_min, depth_max, new_corner));
      // }
      // DLOG(INFO) << "Number of initialized seeds = " << depth_filter_->n_seeds();

      auto img = frame_img.clone();
      cv::cvtColor(img, img, CV_GRAY2BGR);
      if (depth_filter_->options_.seeds_init_type_ == SeedsInitType::FAST_DETECTOR) {
        auto seeds = depth_filter_->get_seeds();
        std::for_each(seeds.begin(), seeds.end(), [&](Seed &seed) {
          Vector2d px = seed.corner_->px_;
          cv::circle(img, cv::Point2f(px.x(), px.y()), 1, cv::Scalar(0, 255, 0), 1);
        });
      } else {
        std::for_each(gt_kps_.begin(), gt_kps_.end(), [&](tuple<float, cv::Point2f> &pt) {
          cv::circle(img, std::get<1>(pt), 1, cv::Scalar(0, 255, 0), 1);
        });
      }
      cv::imshow("img", img);
      cv::waitKey(0);
      continue;
    }

    // Now update the initialized seeds
    frame_cur_.reset(new Frame(i, frame_img, camera_, frame_ts));
    frame_cur_->set_pose(T_w_f);
    depth_filter_->add_frame(frame_cur_);
  }  
}
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]); 

  size_t ref_idx = 0;
  DepthFilterTest filter(ref_idx);
  filter.run();
}
