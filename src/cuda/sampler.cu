#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "checks.cuh"
#include "config.hpp"
#include "sampler.cuh"
#include "utils.hpp"

template <typename T>
__global__ void uniformSamplerKernel(int N, T *x, double lower, double upper,
                                     unsigned long long seed) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double r;
  for (int i = gid; i < N; i += stride) {
    curandState state;
    curand_init(seed, gid, 0, &state);
    r = curand_uniform_double(&state);
    x[i] = static_cast<T>(lower + r * (upper - lower));
  }
}

template <typename T>
__global__ void constantSamplerKernel(int N, T *x, double constant) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double r;
  for (int i = gid; i < N; i += stride) {
    x[i] = static_cast<T>(constant);
  }
}

template <typename T>
void initializeVector(const int N, T *x, Distribution dtype,
                      unsigned long long seed) {
  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);

  if (dtype == Normal) {
    std::runtime_error("Not implemented");
  } else if (dtype == ZeroOne) {
    double lower = 0.0;
    double upper = 1.0;
    uniformSamplerKernel<<<gridDim, blockDim>>>(N, x, lower, upper, seed);
    cudaCheck(cudaGetLastError());
    /* cudaDeviceSynchronize(); */
  } else if (dtype == MinusOnePlusOne) {
    double lower = -1.0;
    double upper = 1.0;
    uniformSamplerKernel<<<gridDim, blockDim>>>(N, x, lower, upper, seed);
    cudaCheck(cudaGetLastError());
    /* cudaDeviceSynchronize(); */
  } else if (dtype == PowTwo) {
    double K = config::K;
    double lower = pow(2.0, K);
    double upper = pow(2.0, K + 1);
    uniformSamplerKernel<<<gridDim, blockDim>>>(N, x, lower, upper, seed);
    cudaCheck(cudaGetLastError());
    /* cudaDeviceSynchronize(); */
  } else if (dtype == Ones) {
    constantSamplerKernel<<<gridDim, blockDim>>>(N, x, 1.0);
    cudaCheck(cudaGetLastError());
  } else {
    std::invalid_argument("Invalid distribution");
  }
}

// Compile templates
template void initializeVector<half>(int N, half *, Distribution,
                                     unsigned long long);
template void initializeVector<float>(int N, float *, Distribution,
                                      unsigned long long);
template void initializeVector<double>(int N, double *, Distribution,
                                       unsigned long long);
