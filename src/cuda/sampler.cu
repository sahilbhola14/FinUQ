#include <cmath>
#include <iostream>
#include <stdexcept>

#include "checks.cuh"
#include "config.hpp"
#include "sampler.cuh"
#include "utils.hpp"

__device__ double sampleRelativeErrorKernel(double urd, curandState *state) {
  double r;
  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  r = curand_uniform_double(state);
  return 2.0 * r * urd - urd;
}

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
__global__ void NormalSamplerKernel(int N, T *x, unsigned long long seed) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double r;
  for (int i = gid; i < N; i += stride) {
    curandState state;
    curand_init(seed, gid, 0, &state);
    r = curand_normal(&state);
    x[i] = static_cast<T>(r);
  }
}

template <typename T>
__global__ void constantSamplerKernel(int N, T *x, double constant) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double r;
  if (gid < N) {
    x[gid] = static_cast<T>(constant);
  } else {
    x[gid] = static_cast<T>(0.0);
  }
}

void getODEParameters(const int N, half *parameters, unsigned long long seed) {
  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  double lower, upper;
  for (int ii = 0; ii < N; ii++) {
    parameters[ii] = static_cast<half>(3.0);
    parameters[N + ii] = static_cast<half>(4.0);
  }
  /* // Sample theta 1 */
  /* lower = 0.1; */
  /* upper = 1.1; */
  /* uniformSamplerKernel<<<gridDim, blockDim>>>(N, parameters, lower, upper,
   * seed); */
  /* cudaCheck(cudaGetLastError()); */
  /* // Sample theta 2 */
  /* lower = 1.0; */
  /* upper = 2.0; */
  /* uniformSamplerKernel<<<gridDim, blockDim>>>(N, parameters + N, lower,
   * upper, seed); */
  /* cudaCheck(cudaGetLastError()); */
}

template <typename T>
void initializeVector(const int N, T *x, Distribution dtype,
                      unsigned long long seed) {
  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);

  if (dtype == Normal) {
    NormalSamplerKernel<<<gridDim, blockDim>>>(N, x, seed);
    cudaCheck(cudaGetLastError());
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
