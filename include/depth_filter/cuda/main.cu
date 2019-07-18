#include "dev_img.cuh"

using namespace learn;
using namespace std;

int main() {
  float data[] = {1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                12, 13, 14, 15};

  AbstractAllocator<float> a(4, 4);
  a.set_data(data);

  float dev2host[16];
  a.get_data(dev2host);
  for (size_t i = 0; i < 16; ++i)
    printf("i = %lu\t:\t%f\n", i, dev2host[i]);
  printf("\n");

  dim3 grid_size((4+16-1)/4, (4+16-1)/4);
  dim3 block_size(16, 16);
  // learn::test_2d<<<grid_size, block_size>>>(a.data_, a.stride_, a.width_, a.height_);
  learn::test_2d_class<<<grid_size, block_size>>>(a.ptr_);
  cudaDeviceSynchronize();

  a.get_data(dev2host);
  for (size_t i = 0; i < 16; ++i)
    printf("i = %lu\t:\t%f\n", i, dev2host[i]);
  printf("\n");
}

