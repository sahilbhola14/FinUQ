#include <cuda_fp16.h>

#include <cassert>
#include <iostream>

#include "checks.cuh"
#include "config.hpp"
#include "definition.hpp"
#include "dotProduct.cuh"
#include "sampler.cuh"
#include "utils.hpp"

template <typename T>
__global__ void dotProductKernel(int N, T *x, T *y, T *result) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
}

void launchDotProductExperiment() {
  double *x_double, *y_double, *partial_double, *result_double;
  float *x_float, *y_float, *partial_float, *result_float;
  half *x_half, *y_half, *partial_half, *result_half;

  const int N = config::N;  // Vector size
  const int K = config::K;  // Sampling interval for PowTwo distribution
  Distribution dtype = config::distType;

  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  unsigned long long seed = 123456789ULL;  // Seed for sampling

  printf("----- config -----\n");
  printf("Vector size: %d\n", N);
  printf("blockDim: %d gridDim: %d\n", blockDim.x, gridDim.x);
  printf("------------------\n");
  assert(N <= blockDim.x * gridDim.x &&
         "Currently only single reduction is implemented");

  // Allocate memory
  cudaCheck(cudaMallocManaged(&x_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&partial_double, gridDim.x * sizeof(double)));

  cudaCheck(cudaMallocManaged(&x_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&y_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&partial_float, gridDim.x * sizeof(float)));

  cudaCheck(cudaMallocManaged(&x_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&y_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&partial_half, gridDim.x * sizeof(half)));

  // Sample (in the lowest precision to avoid representation error)
  initializeVector(N, x_half, dtype, seed);

  // Free memory
  cudaFree(x_double);
  cudaFree(y_double);
  cudaFree(partial_double);
  cudaFree(result_double);

  cudaFree(x_float);
  cudaFree(y_float);
  cudaFree(partial_float);
  cudaFree(result_float);

  cudaFree(x_half);
  cudaFree(y_half);
  cudaFree(partial_half);
  cudaFree(result_half);
}
