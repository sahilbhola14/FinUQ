#include <cuda_fp16.h>

#include <iostream>

#include "addition.cuh"
#include "checks.cuh"
#include "config.hpp"
#include "copy.cuh"
#include "utils.hpp"

template <typename T>
__global__ void sumReduce(int N, T *partial, T *result, Precision prec) {
  extern __shared__ char shared_mem[];
  T *cache = reinterpret_cast<T *>(shared_mem);
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  // Load data to __shared__
  if (gid < N) {
    cache[tid] = partial[gid];
  } else {
    cache[tid] = static_cast<T>(0.0);
  }
  __syncthreads();

  // Reduce
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      if (prec == Double) {
        cache[tid] = __dadd_rn(cache[tid], cache[tid + s]);

      } else if (prec == Float) {
        cache[tid] = __fadd_rn(cache[tid], cache[tid + s]);

      } else if (prec == Half) {
        cache[tid] = __hadd_rn(static_cast<half>(cache[tid]),
                               static_cast<half>(cache[tid + s]));

      } else {
        printf("Error: Invalid Working precision");
        return;
      }
    }
    __syncthreads();
  }

  // Store the sum
  if (tid == 0) result[blockIdx.x] = cache[0];
}

template <typename T>
void allSumReduce(int N, T *partial, T *result, Precision prec) {
  T *post_r, *pre_r;
  int size = N;
  int oldGridDim = N;
  int shared_mem;
  dim3 blockDim = config::blockSize;
  dim3 gridDim;
  gridDim.x = getGridSize(blockDim.x, N);

  // Allocate memory
  cudaCheck(cudaMallocManaged(&pre_r, N * sizeof(T)));
  cudaCheck(cudaMallocManaged(&post_r, gridDim.x * sizeof(T)));

  // Copy the partial to pre_r
  copyVector(N, partial, pre_r);
  cudaCheck(cudaGetLastError());
  while (gridDim.x >= 1) {
    shared_mem = blockDim.x * sizeof(T);
    sumReduce<<<gridDim.x, blockDim.x, shared_mem>>>(
        size, pre_r, post_r, prec);  // Perform reduction
    // Copy values
    cudaDeviceSynchronize();  // Ensure all computations are done
    cudaFree(pre_r);          // Free the memory first
    cudaCheck(cudaMallocManaged(&pre_r, gridDim.x * sizeof(T)));  // New memory
    size = gridDim.x;  // New pre_r size
    copyVector(gridDim.x, post_r, pre_r);
    // Free post
    cudaFree(post_r);
    gridDim.x = getGridSize(blockDim.x, gridDim.x);  // New grid Size
    if (oldGridDim == gridDim.x) break;
    cudaCheck(cudaMallocManaged(&post_r, gridDim.x * sizeof(T)));  // New memory
    oldGridDim = size;
  }
  cudaDeviceSynchronize();
  *result = *pre_r;  // Copy the final result

  cudaFree(pre_r);
}

// template compilation
template void allSumReduce<double>(int, double *, double *, Precision);
template void allSumReduce<float>(int, float *, float *, Precision);
template void allSumReduce<half>(int, half *, half *, Precision);
