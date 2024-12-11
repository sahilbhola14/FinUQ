#include <cuda_fp16.h>

#include <cassert>
#include <iomanip>
#include <iostream>

#include "addition.cuh"
#include "checks.cuh"
#include "config.hpp"
#include "convert.cuh"
#include "definition.hpp"
#include "dotProduct.cuh"
#include "sampler.cuh"
#include "utils.hpp"

template <typename T>
__global__ void dotProductKernel(int N, T *x, T *y, T *result, Precision prec) {
  extern __shared__ char shared_mem[];
  T *cache = reinterpret_cast<T *>(shared_mem);
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  T summation = static_cast<T>(0.0);
  for (int i = gid; i < N; i += stride) {
    if (prec == Double) {
      summation += __dadd_rn(summation, __dmul_rn(x[gid], y[gid]));
    } else if (prec == Float) {
      summation = __fadd_rn(summation, __fmul_rn(x[gid], y[gid]));
    } else if (prec == Half) {
      summation = __hadd_rn(summation, __hmul_rn(static_cast<half>(x[gid]),
                                                 static_cast<half>(y[gid])));
    } else {
      printf("invalid_argument: Invalid precision\n");
      return;
    }
  }
  // Store the summation to __shared__
  if (gid < N) {
    cache[tid] = summation;
  } else {
    cache[tid] = static_cast<T>(0.0);
  }
  __syncthreads();
  // Compute partial sum

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

void launchDotProductExperiment() {
  double *x_double, *y_double, *partial_double, *result_double;
  float *x_float, *y_float, *partial_float, *result_float;
  half *x_half, *y_half, *partial_half, *result_half;

  const int N = config::N;  // Vector size
  Distribution dtype = config::distType;

  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  unsigned long long seed = 123456789ULL;  // Seed for sampling
  /* int cacheSize; */

  printf("----- config -----\n");
  printf("Vector size: %d\n", N);
  printf("blockDim: %d gridDim: %d\n", blockDim.x, gridDim.x);
  printf("------------------\n");

  // Allocate memory
  cudaCheck(cudaMallocManaged(&x_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&partial_double, gridDim.x * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double, sizeof(double)));

  cudaCheck(cudaMallocManaged(&x_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&y_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&partial_float, gridDim.x * sizeof(float)));
  cudaCheck(cudaMallocManaged(&result_float, sizeof(float)));

  cudaCheck(cudaMallocManaged(&x_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&y_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&partial_half, gridDim.x * sizeof(half)));
  cudaCheck(cudaMallocManaged(&result_half, sizeof(half)));

  // Sample (in the lowest precision to avoid representation error)
  initializeVector(N, x_half, dtype, seed);
  initializeVector(N, y_half, dtype, seed + 34577ULL);

  convertHalfToDouble(N, x_half, x_double);
  convertHalfToDouble(N, y_half, y_double);

  convertHalfToFloat(N, x_half, x_float);
  convertHalfToFloat(N, y_half, y_float);

  // Double precision Dot product
  int cacheSize = blockDim.x * sizeof(double);
  dotProductKernel<<<gridDim, blockDim, cacheSize>>>(N, x_double, y_double,
                                                     partial_double, Double);
  cudaCheck(cudaGetLastError());
  allSumReduce(gridDim.x, partial_double, result_double, Double);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();

  // Single precision Dot product
  cacheSize = blockDim.x * sizeof(float);
  dotProductKernel<<<gridDim, blockDim, cacheSize>>>(N, x_float, y_float,
                                                     partial_float, Float);
  cudaCheck(cudaGetLastError());
  allSumReduce(gridDim.x, partial_float, result_float, Float);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();

  // Half precision Dot product
  cacheSize = blockDim.x * sizeof(half);
  dotProductKernel<<<gridDim, blockDim, cacheSize>>>(N, x_half, y_half,
                                                     partial_half, Half);
  cudaCheck(cudaGetLastError());
  allSumReduce(gridDim.x, partial_half, result_half, Half);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  std::cout << std::scientific << std::setprecision(15) << *result_float << ", "
            << *result_double << ", " << static_cast<double>(*result_half)
            << std::endl;

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
