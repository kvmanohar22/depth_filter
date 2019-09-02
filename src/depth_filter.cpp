#include "depth_filter/depth_filter.h"
#include "depth_filter/utils.h"
#include "depth_filter/cameras/pinhole.hpp"

#include <random>
#include <boost/math/distributions/normal.hpp>


namespace df {
size_t df::Seed::seed_counter = 0;

DepthFilter::DepthFilter() {
  LOG(INFO) << "DepthFilter initialized...";
}

void DepthFilter::add_keyframe(FramePtr& frame) {
  DLOG(INFO) << "New KeyFrame added\n";
  frame_ref_ = frame;
  if (options_.seeds_init_type_ == SeedsInitType::FAST_DETECTOR)
    initialize_seeds();
}

void DepthFilter::add_frame(FramePtr& frame) {
  update_seeds(frame);
}

void DepthFilter::initialize_seeds() {

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
  float one_px_angle = one_pixel_angle();
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

    // check along the epipolar line and find a maximum NCC patch
    // Refer: Video-based, Real-Time Multi View Stereo; Section 4.2
    float d_min = max(itr->mu_ - 2*sqrt(itr->sigma2_), 0.000001f);
    float d_max = min(itr->mu_ + 2*sqrt(itr->sigma2_), 100000.0f);
    float d_new; // This depth corresponds to maximum NCC along epipolar line
    if (!find_match_along_epipolar(itr->corner_->frame_, frame, itr->corner_, itr->mu_, d_min, d_max, d_new)) {
      itr->b_++;
      ++itr;
      continue;
    }

    Vector3d rp = itr->corner_->f_ * d_new;
    float tau = compute_tau(rp, T_cur_ref.inverse().translation(), itr->corner_->f_, one_px_angle);

    update_seed(&*itr, d_new, tau*tau);
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
  // TODO: Check for corner cases
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

float DepthFilter::one_pixel_angle() {
  double f = dynamic_cast<df::Pinhole*>(frame_ref_->cam_)->fx();
  return 2.0 * atan(1.0f / (2 * f));
}

// Reference: https://kvmanohar22.github.io/depth-estimation/#bayesian_variance
float DepthFilter::compute_tau(Vector3d rp, Vector3d t, Vector3d f, float one_px_angle) {
  Vector3d a = rp - t;
  double t_norm = t.norm();
  double alpha = acos(f.dot(t) / t_norm);
  double beta = acos(-t.dot(a) / (t_norm * a.norm()));
  double beta_plus = beta + one_px_angle;
  double gamma = df::PI - alpha - beta_plus;
  double rp_plus_norm = t_norm * (sin(beta_plus) / sin(gamma));
  double tau = (rp_plus_norm - rp.norm());
  return tau;
}

bool DepthFilter::find_match_along_epipolar(
      FramePtr ref_frame, FramePtr cur_frame,
      Corner* corner,
      float d_current, 
      float d_min, float d_max, float d_new) {
  
  Sophus::SE3 T_ref_cur = ref_frame->T_f_w_ * cur_frame->T_f_w_.inverse();
  Vector2d pt_min = utils::project2d(T_ref_cur.inverse() * (corner->f_ * d_min)); 
  Vector2d pt_max = utils::project2d(T_ref_cur.inverse() * (corner->f_ * d_max)); 

  Vector2d px_min(cur_frame->cam_->world2cam(pt_min));
  Vector2d px_max(cur_frame->cam_->world2cam(pt_max));

  float line_len = (px_max-px_min).norm();
  size_t n_steps = line_len / 0.7;
  DLOG(INFO) << "#Steps = " << n_steps;

#ifdef DEBUG_YES
  auto px = corner->px_;
  auto img_ref = ref_frame->img_.clone();
  auto img_cur = cur_frame->img_.clone();
  cv::cvtColor(img_ref, img_ref, CV_GRAY2BGR);
  cv::cvtColor(img_cur, img_cur, CV_GRAY2BGR);
  // TODO: Draw this rectangle along epipolar line direction
  cv::line(img_cur, cv::Point2f(px_min.x(), px_min.y()), cv::Point2f(px_max.x(), px_max.y()), cv::Scalar(255, 0, 0), 1, CV_AA);
  cv::rectangle(img_ref, cv::Point2f(px.x()-2, px.y()-2), cv::Point2f(px.x()+2, px.y()+2), cv::Scalar(255, 0, 0), 1, CV_AA);
  cv::circle(img_cur, cv::Point2f(px_min.x(), px_min.y()), 2, cv::Scalar(0, 255, 0), 2);
  cv::circle(img_cur, cv::Point2f(px_max.x(), px_max.y()), 2, cv::Scalar(0, 0, 255), 2);
  cv::imshow("ref image", img_ref);
  cv::imshow("cur image", img_cur);
  cv::waitKey(0);
#endif 

  return false;
}


} // namespace df

