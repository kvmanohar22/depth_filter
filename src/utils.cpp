#include "depth_filter/utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace df {

namespace utils {


float cross_correlation_single_patch(
    float *f, float *g,
    size_t f_cols, size_t start_idx, size_t end_idx) {
  float score=0.0f;
  return score;
}

float cross_correlation(cv::Mat &f, cv::Mat &g) {
  auto f_ptr = f.ptr<float>();
  auto g_ptr = g.ptr<float>();
  size_t stride_r = g.cols;
  size_t stride_c = g.rows;
  size_t stride2 = stride_r*stride_c;
  size_t f_rows = f.rows;
  size_t f_cols = f.cols;
  std::ofstream fii;
  fii.open("data.txt");
  cv::Mat score_img(f_rows-stride_r+1, f_cols-stride_c+1, CV_32F);
  auto score_ptr = score_img.ptr<float>();
  for (size_t r=0; r<f_rows; ++r) {
    for (size_t c=0; c<f_cols; ++c) {
      float score=0;
      if (r+stride_r<=f_rows && c+stride_c<=f_cols) {
        for(size_t i=0; i<stride_r;++i) {
          for(size_t j=0; j<stride_c;++j) {
            auto f_val = *(f_ptr+(r+i)*f_cols+c+j);
            auto g_val = *(g_ptr+i*stride_r+j);
            score += f_val * g_val;
          }
        }
        *score_ptr = score;
        ++score_ptr; 
        fii << score << " ";
      }
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
      stddev += (*img_ptr - mean)*(*img_ptr-mean);
  stddev = sqrt(stddev);
  img_ptr = img.ptr<float>();
  for (size_t c=0;c<img.cols;++c)
  for (size_t c=0;c<img.cols;++c)
    for (size_t r=0;r<img.rows;++r, ++img_ptr)
      *img_ptr = (*img_ptr - mean)/stddev;
}

float normalized_cross_correlation(cv::Mat &f, cv::Mat &g) {
  assert((f.type() == CV_32F) && (g.type() == CV_32F));
  float f_mean=0.0f, g_mean=0.0f;
  auto f_ptr = f.ptr<float>();
  auto g_ptr = g.ptr<float>();
  for (size_t c=0;c<f.cols;++c)
    for (size_t r=0;r<f.rows;++r, ++f_ptr)
      f_mean += *f_ptr;
  for (size_t c=0;c<g.cols;++c)
    for (size_t r=0;r<g.rows;++r, ++g_ptr)
      g_mean += *g_ptr;
  f_mean /= (f.rows*f.cols);
  g_mean /= (g.rows*g.cols);
 
  float f_std=0.0f, g_std=0.0f;
  f_ptr = f.ptr<float>();
  g_ptr = g.ptr<float>();
  for (size_t c=0;c<f.cols;++c)
    for (size_t r=0;r<f.rows;++r, ++f_ptr)
      f_std += (*f_ptr - f_mean)*(*f_ptr-f_mean);
  for (size_t c=0;c<g.cols;++c)
    for (size_t r=0;r<g.rows;++r, ++g_ptr)
      g_std += (*g_ptr - g_mean)*(*g_ptr-g_mean);
  f_std = sqrt(f_std);
  g_std = sqrt(g_std);

  f_ptr = f.ptr<float>();
  g_ptr = g.ptr<float>();
  for (size_t c=0;c<f.cols;++c)
    for (size_t r=0;r<f.rows;++r, ++f_ptr)
      *f_ptr = (*f_ptr - f_mean)/f_std;
  for (size_t c=0;c<g.cols;++c)
    for (size_t r=0;r<g.rows;++r, ++g_ptr)
      *g_ptr = (*g_ptr - g_mean)/g_std;

  return cross_correlation(f, g);
}

} // namespace utils

} // namespace df

