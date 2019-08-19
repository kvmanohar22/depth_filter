#include "depth_filter/utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace df {

namespace utils {

float cross_correlation_single_patch(
    float *f, float *g,
    size_t f_cols, size_t g_cols,
    size_t start_idx_fx, size_t start_idx_fy,
    size_t start_idx_gx, size_t start_idx_gy,
    size_t tx, size_t ty) {
  float score = 0.0f;
  for (size_t r = 0; r < ty; ++r) {
    for (size_t c = 0; c < tx; ++c) {
      auto f_val = *(f + (start_idx_fy + r) * f_cols + start_idx_fx + c);
      auto g_val = *(g + (start_idx_gy + r) * g_cols + start_idx_gx + c);
      score += f_val * g_val;
    }
  }
  return score;
}

bool cross_correlation(cv::Mat &f, cv::Mat &g) {
  auto f_ptr = f.ptr<float>();
  auto g_ptr = g.ptr<float>();
  size_t stride_r = g.cols;
  size_t stride_c = g.rows;
  size_t stride2 = stride_r*stride_c;
  size_t f_rows = f.rows;
  size_t f_cols = f.cols;
  cv::Mat score_img(f_rows-stride_r+1, f_cols-stride_c+1, CV_32F);
  auto score_ptr = score_img.ptr<float>();
  for (size_t r=0; r<f_rows; ++r) {
    for (size_t c=0; c<f_cols; ++c) {
      auto score = cross_correlation_single_patch(
          f_ptr, g_ptr, f_cols, stride_r, c, r, 0, 0, stride_r, stride_c);
        *score_ptr = score;
        ++score_ptr; 
      }
    }
  return 0;
}

void normalize_image(cv::Mat &img) {
  assert(img.type() == CV_32F);
  float mean = 0.0f, stddev = 0.0f;
  auto img_ptr = img.ptr<float>();
  for (size_t r = 0; r < img.rows; ++r)
    for (size_t c = 0; c < img.cols; ++c, ++img_ptr)
      mean += *img_ptr;
  mean /= (img.rows*img.cols);
  img_ptr = img.ptr<float>();
  for (size_t c=0;c<img.cols;++c)
    for (size_t r=0;r<img.rows;++r, ++img_ptr)
      stddev += (*img_ptr - mean)*(*img_ptr - mean);
  stddev = sqrt(stddev);
  img_ptr = img.ptr<float>();
  for (size_t c=0;c<img.cols;++c)
    for (size_t r=0;r<img.rows;++r, ++img_ptr)
      *img_ptr = (*img_ptr - mean)/stddev;
}

void normalized_cross_correlation(cv::Mat &f, cv::Mat &g) {
  normalize_image(f);
  normalize_image(g);
  cross_correlation(f, g);
}

} // namespace utils

} // namespace df

