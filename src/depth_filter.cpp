#include "depth_filter/depth_filter.h"
#include "depth_filter/utils.h"
#include "depth_filter/feature_detector.h"
#include "depth_filter/cameras/pinhole.hpp"

#include <random>
#include <boost/math/distributions/normal.hpp>


namespace df {
size_t df::Seed::seed_counter = 0;

DepthFilter::DepthFilter(AbstractCamera *cam) {
  LOG(INFO) << "DepthFilter initialized...";

  float focal_len = dynamic_cast<df::Pinhole*>(cam)->fx();
  one_px_angle_ = 2.0 * atan(1.0f / (2 * focal_len));
}

void DepthFilter::add_keyframe(FramePtr& frame, 
  float depth_min, float depth_mean, float depth_max) {
  DLOG(INFO) << "New KeyFrame added\n";
  frame_ref_ = frame;
  if (options_.seeds_init_type_ == SeedsInitType::FAST_DETECTOR)
    initialize_seeds(depth_min, depth_mean, depth_max);
}

void DepthFilter::add_frame(FramePtr& frame) {
  update_seeds(frame);
}

void DepthFilter::initialize_seeds(
  float depth_min, float depth_mean, float depth_max)
{
  list<Corner*> new_corners;
  FastDetector::detect(frame_ref_->img_, new_corners);
  std::for_each(new_corners.begin(), new_corners.end(), [&](Corner* crn) {
    seeds_.emplace_back(Seed(depth_mean, depth_min, depth_max, crn));
  });
  DLOG(INFO) << "Number of initialized seeds with fast-corner = " << seeds_.size();
}

list<Seed>& DepthFilter::get_mutable_seeds() {
  return seeds_;
}

void DepthFilter::update_seeds_loop() {
  /// main outer loop for updating seeds
  while (true) {

  }
}

void DepthFilter::update_seeds(FramePtr& frame) {
  DLOG(INFO) << "New frame added for updating seeds...";
  list<Seed>::iterator itr = seeds_.begin();
  for (; itr != seeds_.end();) {
    auto T_cur_ref = frame->T_f_w_ * frame_ref_->T_f_w_.inverse();
    const Vector3d xyz_f(T_cur_ref * (itr->mu_ * itr->corner_->f_));

    // point is behind the camera
    if (xyz_f.z() < 0) {
      ++itr;
      continue;
    }

    // point does not lie within the image 
    if (!frame->cam_->is_in_frame(frame->cam_->world2cam(xyz_f).cast<int>())) {
      ++itr;
      continue;
    }

    float d_min = max(itr->mu_ - 1*sqrt(itr->sigma2_), 0.000001f);
    float d_max = min(itr->mu_ + 1*sqrt(itr->sigma2_), 100000.0f);
    float d_new;
    if (!find_match_along_epipolar(itr->corner_->frame_, frame, itr->corner_, itr->mu_, d_min, d_max, d_new)) {
      itr->b_++;
      ++itr;
      continue;
    }

    Vector3d rp = itr->corner_->f_ * d_new;
    float tau = compute_tau(rp, T_cur_ref.inverse().translation(), itr->corner_->f_);

    update_seed(&*itr, d_new, tau*tau);
    
    if (itr->sigma2_ < options_.sigma2_convergence_thresh_) {
      assert(itr->corner_->xyz_ == nullptr);
      Vector3d xyz_world(itr->corner_->frame_->T_f_w_.inverse() * (itr->corner_->f_ * itr->mu_));
      Point* point = new Point(xyz_world, itr->corner_);
      itr->corner_->xyz_ = point;
      LOG(INFO) << "Converged seed id = " << itr->idx_;
    }
  }
}

void DepthFilter::update_seed(Seed* seed, float d_mu, float d_tau2) {
  float a = seed->a_;
  float b = seed->b_;
  float mu = seed->mu_;
  float sigma2 = seed->sigma2_;
  float z_range = seed->z_range_;

  float s2 = 1.0/(1.0/sigma2 + 1.0/d_tau2);
  float m = s2 * (1.0/d_tau2 + mu/sigma2);

  // std::normal_distribution<float> norm_dist(mu, sigma2+new_tau2);
  // float c1 = (a / (a+b)) * norm_dist(new_mu);
  float scale = sqrt(sigma2+d_tau2);
  if (std::isnan(scale)) {
    LOG(WARNING) << "Nan encountered in seeds udpating";
  }
  boost::math::normal_distribution<float> nd(mu, scale);
  float c1 = a * boost::math::pdf(nd, d_mu) / (a + b);
  float c2 = b / (z_range * (a + b)); 

  float new_mu = c1 * m + c2 * mu;
  float new_tau2 = c1 * (s2 + m*m) + c2 * (mu*mu + sigma2) - new_mu * new_mu;

  float g = c1*(a+1.0)/(a+b+1.0) + c2*a/(a+b+1.0);
  float h = c1*(a+1.0)*(a+2.0)/((a+b+1.0)*(a+b+2.0))+c2*a*(a+1.0)/((a+b+1.0)*(a+b+2.0));
  
  float new_a = (h-g)/(g-h/g);
  float new_b = a*(1.0-g)/g;

  // update the parameters
  seed->a_ = new_a;
  seed->b_ = new_b;
  seed->mu_ = new_mu;
  seed->sigma2_ = new_tau2;
}

// Reference: https://kvmanohar22.github.io/depth-estimation/#bayesian_variance
float DepthFilter::compute_tau(Vector3d rp, Vector3d t, Vector3d f) {
  Vector3d a = rp - t;
  double t_norm = t.norm();
  double alpha = acos(f.dot(t) / t_norm);
  double beta = acos(-t.dot(a) / (t_norm * a.norm()));
  double beta_plus = beta + one_px_angle_;
  double gamma = df::PI - alpha - beta_plus;
  double rp_plus_norm = t_norm * (sin(beta_plus) / sin(gamma));
  double tau = (rp_plus_norm - rp.norm());
  return tau;
}

bool DepthFilter::find_match_along_epipolar(
      FramePtr ref_frame, FramePtr cur_frame,
      Corner* corner, float d_current,
      float d_min, float d_max, float &d_new)
{
  Sophus::SE3 T_ref_cur = ref_frame->T_f_w_ * cur_frame->T_f_w_.inverse();
  Vector2d pt_min = utils::project2d(T_ref_cur.inverse() * (corner->f_ * d_min)); 
  Vector2d pt_max = utils::project2d(T_ref_cur.inverse() * (corner->f_ * d_max)); 
  Vector2d line_dir = pt_min - pt_max;

  Vector2d px_min(cur_frame->cam_->world2cam(pt_min));
  Vector2d px_max(cur_frame->cam_->world2cam(pt_max));

  float line_len = (px_max-px_min).norm();
  size_t n_steps = line_len / 0.7; // TODO: Why 0.7?
  Vector2d step = line_dir / n_steps;
  if (n_steps > options_.max_steps_) {
    LOG(WARNING) << "No. of steps = " << n_steps << " exceeded limit = " << options_.max_steps_;
    return false;
  }

  // pre-compute the reference patch
  uint8_t ref_patch[options_.patch_size_*options_.patch_size_] __attribute__ ((aligned(16)));
  create_ref_patch(ref_patch, corner->px_, frame_ref_->img_);
  PatchScore patch_score(ref_patch);
  int zmssd_best = PatchScore::threshold();

  // go along the epipolar line segment and find the patch with min ZMSSD
  Vector2d uv = pt_max-step;
  Vector2d uv_best;
  Vector2i last_px(0, 0);
  for (size_t i=0; i<n_steps; ++i, uv+=step) {
    Vector2d px(ref_frame->cam_->world2cam(uv));
    Vector2i px_i(px[0]+0.5, px[1]+0.5);
    
    if (px_i == last_px) {
      continue;
    }
    last_px = px_i;

    if (!cur_frame->cam_->is_in_frame(px_i, options_.patch_size_)) {
      continue;
    }

    uint8_t* cur_patch_ptr = cur_frame->img_.data
                             + (px_i[1]-options_.half_patch_size_)*cur_frame->img_.cols
                             + (px_i[0]-options_.half_patch_size_);
    int zmssd = patch_score.computeScore(cur_patch_ptr, cur_frame->img_.cols);
    if (zmssd < zmssd_best) {
      zmssd_best = zmssd;
      uv_best = uv;
    }
    if (options_.verbose_) {
      cout << "iter = " << i << " / " << n_steps << " score = " << zmssd << " best = " << zmssd_best << endl;

      auto px = corner->px_;
      auto img_ref = ref_frame->img_.clone();
      auto img_cur = cur_frame->img_.clone();
      cv::cvtColor(img_ref, img_ref, CV_GRAY2BGR);
      cv::cvtColor(img_cur, img_cur, CV_GRAY2BGR);
      auto uv_best_px = ref_frame->cam_->world2cam(uv_best);
      auto uv_cur_px =  ref_frame->cam_->world2cam(uv);
      cv::line(img_cur, cv::Point2f(px_min.x(), px_min.y()), cv::Point2f(px_max.x(), px_max.y()), cv::Scalar(255, 0, 0), 1, CV_AA);
      cv::rectangle(img_ref, cv::Point2f(px.x()-2, px.y()-2), cv::Point2f(px.x()+2, px.y()+2), cv::Scalar(255, 0, 0), 1, CV_AA);
      // cv::circle(img_cur, cv::Point2f(px_min.x(), px_min.y()), 2, cv::Scalar(0, 255, 0), 2);
      // cv::circle(img_cur, cv::Point2f(px_max.x(), px_max.y()), 2, cv::Scalar(0, 0, 255), 2);
      cv::circle(img_cur, cv::Point2f(uv_best_px.x(), uv_best_px.y()), 1, cv::Scalar(0, 0, 255), 1);
      // cv::circle(img_cur, cv::Point2f(uv_cur_px.x(), uv_cur_px.y()), 2, cv::Scalar(255, 255, 0), 2);
      cv::imshow("ref image", img_ref);
      cv::imshow("cur image", img_cur);
      cv::waitKey(0);
    }
  }

  if (zmssd_best < PatchScore::threshold()) {
    Vector2d px_cur = cur_frame->cam_->world2cam(uv_best);
    if(triangulate(T_ref_cur.inverse(), corner->f_, utils::unproject2d(uv_best).normalized(), d_new))
      return true;
    return false;
  }
  return false;
}

void DepthFilter::create_ref_patch(uint8_t* patch, Vector2d& px_ref, cv::Mat& img_ref) {
  uint8_t* patch_ptr = patch;
  const Vector2f px_ref_f = px_ref.cast<float>();
  for (int y=0; y<options_.patch_size_; ++y) {
    for (int x=0; x<options_.patch_size_; ++x, ++patch_ptr) {
      Vector2f px_patch(x-options_.half_patch_size_, y-options_.half_patch_size_);
      const Vector2f px(px_patch + px_ref_f);
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        *patch_ptr = 0;
      else
        *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

bool DepthFilter::triangulate(
    const Sophus::SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    float& depth)
{
  Matrix<double,3,2> A; A << T_search_ref.rotation_matrix() * f_ref, f_cur;
  const Matrix2d AtA = A.transpose()*A;
  if(AtA.determinant() < 0.000001)
    return false;
  const Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.translation();
  depth = fabs(depth2[0]);
  return true;
}


} // namespace df

