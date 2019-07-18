#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H

#include "global.h"
#include <stdio.h>

namespace depth_filter {

template <typename T>
__global__
void test_2d(T* data, size_t stride, size_t cols, size_t rows) {
  size_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
  size_t tidy = blockIdx.y*blockDim.y+threadIdx.y;
  if ((tidx < cols) && (tidy < rows))
    data[tidy*stride+tidx] = data[tidy*stride+tidx]*tidx*tidy;
}

template <typename T>
class AbstractAllocator {
public:
  __host__
  AbstractAllocator(size_t width, size_t height) : width_(width), height_(height) {
    check_cuda_errors(cudaMallocPitch(&data_, &pitch_, width_*sizeof(T), height_));
    stride_ = pitch_ / sizeof(T);
    check_cuda_errors(cudaMallocManaged((void**)&ptr_, sizeof(*this)));  
    check_cuda_errors(cudaMemcpy(ptr_, this, sizeof(*this), cudaMemcpyHostToDevice));
  }

  __device__
  T& operator() (size_t x, size_t y) {
    return at_xy(x, y);
  }
 
  __device__ 
  T& at_xy(size_t x, size_t y) {
     return data_[y*stride_+x];
  }

  __host__
  void set_data(const T* aligned_row_major) {
    check_cuda_errors(cudaMemcpy2D(data_, pitch_, aligned_row_major,
      width_*sizeof(T), width_*sizeof(T), height_, cudaMemcpyHostToDevice));  
  }

  __host__
  void get_data(T* aligned_row_major) {
    check_cuda_errors(cudaMemcpy2D(aligned_row_major, width_*sizeof(T), data_,
      pitch_, width_*sizeof(T), height_, cudaMemcpyDeviceToHost));  
  }

  __host__ __device__ __forceinline__
  size_t width() const { return width_; }

  __host__ __device__ __forceinline__
  size_t height() const { return height_; }

  __host__ __device__ __forceinline__
  size_t stride() const { return stride_; }


  __host__
  ~AbstractAllocator() {
    check_cuda_errors(cudaFree(data_));
    check_cuda_errors(cudaFree(ptr_));
  }

  // params
  size_t      width_;   // width of image 
  size_t      height_;  // height of image 
  size_t      pitch_;   // pitch for cuda memory allocation 
  size_t      stride_;  // stride for next row
  T           *data_;
  AbstractAllocator<T> *ptr_;
};

template <typename T>
__global__
void test_2d_class(AbstractAllocator<T> *dev_ptr) {
  size_t tidx = blockIdx.x*blockDim.x+threadIdx.x;
  size_t tidy = blockIdx.y*blockDim.y+threadIdx.y;
  if ((tidx < dev_ptr->width()) && (tidy < dev_ptr->height()))
    (*dev_ptr)(tidx, tidy) = (*dev_ptr)(tidx, tidy)*tidx*tidy;
}

} // namespace depth_filter

#endif

