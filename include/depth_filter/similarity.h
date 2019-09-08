#ifndef _DF_SIMILARITY_H_
#define _DF_SIMILARITY_H_

#include "depth_filter/global.h"

namespace utils {
namespace similarity {

template <typename DTYPE, size_t N>
class Abstract {
public:
  Abstract()
    : half_patch_size_(N),
      patch_area_(2*N*2*N),
      ref_patch_(nullptr)
  {}

  Abstract(DTYPE* ref_patch)
    : half_patch_size_(N),
      patch_area_(2*N*2*N),
      ref_patch_(ref_patch),
      sumR_(0),
      sumR2_(0)
  {
    if (ref_patch_!=nullptr)
      recompute();
  }
  virtual ~Abstract() =default;
  virtual double similarity(DTYPE* new_patch, size_t cols) const =0;
  inline const size_t half_patch_size() const { return half_patch_size_; }
  inline const size_t patch_area() const      { return patch_area_; }
  inline const double sumR() const { return sumR_; }
  inline const double sumR2() const { return sumR2_; }
  inline const DTYPE* ref_patch() const { return ref_patch_; }
  inline void set_ref_patch(DTYPE* ref_patch) {
    ref_patch_ = ref_patch;
    recompute();
  }
  inline void recompute() {
    sumR_ = 0;
    sumR2_ = 0;
    size_t patch_size = 2*half_patch_size_;
    for(size_t y=0; y<patch_size; ++y) {
      for(size_t x=0; x<patch_size; ++x) {
        DTYPE val = *(ref_patch_+y*patch_size+x);
        sumR_ += val;
        sumR2_ += val*val;
      }
    }
  }

protected:
  DTYPE* ref_patch_;       // reference patch
  size_t half_patch_size_; // rows/2 or cols/2
  size_t patch_area_;      // rows*cols
  double sumR_;            // sum of all pixels
  double sumR2_;           // sum of squares of all pixels
};

/// Sum of Square Differences
template <typename DTYPE, size_t N>
class SSD : public Abstract<DTYPE, N> {
public:
  SSD() : Abstract<DTYPE, N>() {}
  SSD(DTYPE* ref_patch_) : Abstract<DTYPE, N>(ref_patch_) {}
  virtual ~SSD() =default;
  double similarity(DTYPE* new_patch, size_t cols) const override {
    double sumC2=0, sumRC=0;
    const DTYPE* ref_patch=Abstract<DTYPE, N>::ref_patch();
    const size_t patch_size = 2*Abstract<DTYPE, N>::half_patch_size();
    for(size_t y=0; y<patch_size; ++y) {
      for(size_t x=0; x<patch_size; ++x) {
        DTYPE val = *(new_patch+y*cols+x);
        sumC2 += val*val;
        sumRC += val*ref_patch[y*patch_size+x];
      }
    }
    const double sR2=Abstract<DTYPE, N>::sumR2();
    return sR2+sumC2-2*sumRC;
  }
};

/// Zero Mean Sum of Square Differences
template <typename DTYPE, size_t N>
class ZMSSD : public Abstract<DTYPE, N> {
public:
  ZMSSD() : Abstract<DTYPE, N>() {}
  ZMSSD(DTYPE* ref_patch_) : Abstract<DTYPE, N>(ref_patch_) {}
  virtual ~ZMSSD() =default;
  double similarity(DTYPE* new_patch, size_t cols) const override {
    double sumC=0, sumC2=0, sumRC=0;
    const DTYPE* ref_patch=Abstract<DTYPE, N>::ref_patch();
    const size_t patch_size = 2*Abstract<DTYPE, N>::half_patch_size();
    for(size_t y=0; y<patch_size; ++y) {
      for(size_t x=0; x<patch_size; ++x) {
        DTYPE val = *(new_patch+y*cols+x);
        sumC += val;
        sumC2 += val*val;
        sumRC += val*ref_patch[y*patch_size+x];
      }
    }
    const double sR=Abstract<DTYPE, N>::sumR(), sR2=Abstract<DTYPE, N>::sumR2();
    return sR2+sumC2-2*sumRC-(sR*sR+sumC*sumC-2*sR*sumC)/(patch_size*patch_size);
  }
};

/// Normalized Cross Correlation
template <typename DTYPE, size_t N>
class NCC : public Abstract<DTYPE, N> {
public:
  NCC() : Abstract<DTYPE, N>() {}
  NCC(DTYPE* ref_patch_) : Abstract<DTYPE, N>(ref_patch_) {} 
  virtual ~NCC() =default;
  double similarity(DTYPE* new_patch, size_t cols) const override {
    double sumC2=0, sumRC=0;
    const DTYPE* ref_patch=Abstract<DTYPE, N>::ref_patch();
    const size_t patch_size = 2*Abstract<DTYPE, N>::half_patch_size();
    for(size_t y=0; y<patch_size; ++y) {
      for(size_t x=0; x<patch_size; ++x) {
        DTYPE val = *(new_patch+y*cols+x);
        sumC2 += val*val;
        sumRC += val*ref_patch[y*patch_size+x];
      }
    }
    const double sR2=Abstract<DTYPE, N>::sumR2();
    return sumRC / sqrt(sR2*sumC2);
  }
};

/// Zero Mean Normalized Cross Correlation
template <typename DTYPE, size_t N>
class ZMNCC : public Abstract<DTYPE, N> {
public:
  ZMNCC() : Abstract<DTYPE, N>() {}
  ZMNCC(DTYPE* ref_patch_) : Abstract<DTYPE, N>(ref_patch_) {}
  virtual ~ZMNCC() =default;
  double similarity(DTYPE* new_patch, size_t cols) const override {
    double sumC=0, sumC2=0, sumRC=0;
    const DTYPE* ref_patch=Abstract<DTYPE, N>::ref_patch();
    const size_t patch_size = 2*Abstract<DTYPE, N>::half_patch_size();
    for(size_t y=0; y<patch_size; ++y) {
      for(size_t x=0; x<patch_size; ++x) {
        DTYPE val = *(new_patch+y*cols+x);
        sumC += val;
        sumC2 += val*val;
        sumRC += val*ref_patch[y*patch_size+x];
      }
    }
    const size_t full_area = Abstract<DTYPE, N>::patch_area();
    const double sR=Abstract<DTYPE, N>::sumR(), sR2=Abstract<DTYPE, N>::sumR2();
    return (sumRC-sR*sumC/full_area)/sqrt((sR2-sR*sR/full_area)*(sumC2-sumC*sumC/full_area));
  }
};

} // namespace similarity
} // namespace utils

#endif
