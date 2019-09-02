#include "depth_filter/depth_filter.h"
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
    // TODO: Implementation is left
    double z_along_ray_with_max_ncc;
    float d_min = itr->mu_ - sqrt(itr->sigma2_);
    float d_max = itr->mu_ + sqrt(itr->sigma2_);
    float d_new; // This depth corresponds to maximum NCC along epipolar line
    if (!find_match_along_epipolar(itr->corner_->frame_, frame, itr->mu_, d_min, d_max, d_new)) {
      itr->b_++;
      ++itr;
      continue;
    }

    Vector3d rp = itr->corner_->f_ * z_along_ray_with_max_ncc;
    float tau = compute_tau(rp, T_cur_ref.inverse().translation(), itr->corner_->f_, one_px_angle);

    update_seed(&*itr, z_along_ray_with_max_ncc, tau*tau);
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
      FramePtr frame_ref, FramePtr frame_cur,
      float d_current, 
      float d_min, float d_max, float d_new) {

  return false;  
}


} // namespace df

