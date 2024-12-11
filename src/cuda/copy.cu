#include <cuda_fp16.h>

#include "config.hpp"
#include "copy.cuh"
#include "utils.hpp"

template <typename T>
__global__ void copyVectorKernel(int N, T *source, T *target) {
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < N; i += stride) {
    target[i] = source[i];
  }
}

template <typename T>
void copyVector(int N, T *source, T *target) {
  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  copyVectorKernel<<<gridDim, blockDim>>>(N, source, target);
}

// Template compile
template void copyVector<double>(int, double *, double *);
template void copyVector<float>(int, float *, float *);
template void copyVector<half>(int, half *, half *);
