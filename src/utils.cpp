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

  if (start_idx_gx < 0 || start_idx_gx > g_cols)
    return -2.0f;
  if (start_idx_fx < 0 || start_idx_fx > f_cols)
    return -2.0f;

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

cv::Mat normalize_image(cv::Mat &img,
    size_t r_idx, size_t c_idx,
    int cols, int rows)
{
  assert(img.type() == CV_32F);

  if (r_idx < 0 || c_idx < 0 || r_idx > img.rows || c_idx > img.cols) {
    cerr << "WARNING: indices are out of bounds\n";
    return img.clone();
  }

  rows = rows == -1 ? img.rows : rows;
  cols = cols == -1 ? img.cols : cols;

  size_t max_row = r_idx + rows;
  size_t max_col = c_idx + cols;
  auto img_copy = img.clone();
  float mean = 0.0f, stddev = 0.0f;
  auto img_ptr = img_copy.ptr<float>();
  for (size_t r = r_idx; r < max_row; ++r)
    for (size_t c = c_idx; c < max_col; ++c)
      mean += *(img_ptr + r*img.cols+c);
  mean /= (rows*cols);
  for (size_t c=c_idx;c<max_col;++c) {
    for (size_t r=r_idx;r<max_row;++r) {
      auto val = *(img_ptr +r*img.cols+c);
      stddev += (val - mean)*(val - mean);
    }
  }
  stddev = sqrt(stddev) + df::EPS; // to avoid singularities
  for (size_t c=c_idx;c<max_col;++c)
    for (size_t r=r_idx;r<max_row;++r)
      *(img_ptr+r*img.cols+c) = (*(img_ptr+r*img.cols+c)-mean)/stddev;

  return img_copy.clone();
}

void normalized_cross_correlation(cv::Mat &f, cv::Mat &g) {
  normalize_image(f);
  normalize_image(g);
  cross_correlation(f, g);
}

bool load_kitti_velodyne_scan(std::string file, df::PointCloud *cloud) {
  ifstream fpoint;
  fpoint.open(file, ios::binary);
  if (!fpoint.good()) {
    cerr << "Cannot open the file: " << file << endl;
    return false;
  }

  for (size_t i = 0; fpoint.good() && !fpoint.eof(); ++i) {
    Vector3f point; float reflectance;
    float *ptr = point.data();
    fpoint.read((char*)ptr, 3*sizeof(float));
    fpoint.read((char*)&reflectance, sizeof(float));
    cloud->points_.push_back(point);
  }
  return true;
}

} // namespace utils

} // namespace df

