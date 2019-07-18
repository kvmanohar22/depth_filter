#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <iostream>

namespace depth_filter {

#define check_cuda_errors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA Error = " << static_cast<unsigned int>(result) << " @ "
              << file << " : " << line << " '" << func << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}

} // namespace depth_filter

#endif // _GLOBAL_H_

