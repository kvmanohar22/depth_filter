#include "depth_filter/utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace df {

namespace utils {

float cross_correlation(cv::Mat &f, cv::Mat &g) {
  auto f_ptr = f.ptr<float>();
  auto g_ptr = g.ptr<float>();
  size_t stride_r = g.cols;
  size_t stride_c = g.rows;
  size_t stride2 = stride_r*stride_c;
  size_t f_rows = f.rows;
  size_t f_cols = f.cols;
  cv::Mat score_img(f_rows/stride_r, f_cols/stride_c, CV_32F);
  auto score_ptr = score_img.ptr<float>();
  for (size_t r=0; r<f_rows; ++r) {
    for (size_t c=0; c<f_cols; ++c) {
      float score=0;
      if (r+stride_r<f_rows && c+stride_c<f_cols) {
        g_ptr = g.ptr<float>();
        for(size_t i=0; i<stride_r;++i)
          for(size_t j=0; j<stride_c;++j, ++g_ptr)
            score += (*(g_ptr+i*stride_r+j) * (*(f_ptr+(r+i)*f_rows+c+j)));
      }
     *score_ptr = score; 
    }
  }
  cv::imwrite("score.png", score_img);
  return 0;
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
      *g_ptr -= (*g_ptr - g_mean)/g_std;
  return cross_correlation(f, g);
}

} // namespace utils

} // namespace df

