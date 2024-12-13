#include <cuda_fp16.h>

#include "config.hpp"
#include "copy.cuh"
#include "utils.hpp"

template <typename T>
__global__ void copyVectorKernel(int N, T *source, T *target, bool abs) {
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < N; i += stride) {
    if (abs == true) {
      if constexpr (std::is_same<T, float>::value) {
        target[i] = fabsf(source[i]);
      } else if constexpr (std::is_same<T, double>::value) {
        target[i] = fabs(source[i]);
      } else if constexpr (std::is_same<T, half>::value) {
        target[i] = __habs(source[i]);
      } else {
        printf("Invalid data type");
        return;
      }
    } else {
      target[i] = source[i];
    }
  }
}

template <typename T>
void copyVector(int N, T *source, T *target, bool abs) {
  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  copyVectorKernel<<<gridDim, blockDim>>>(N, source, target, abs);
}

// Template compile
template void copyVector<double>(int, double *, double *, bool);
template void copyVector<float>(int, float *, float *, bool);
template void copyVector<half>(int, half *, half *, bool);
