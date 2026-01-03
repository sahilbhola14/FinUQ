#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <iostream>

#include "definition.hpp"
#include "dot_product_cuda.cuh"
#include "rounding_error_model.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* sequential dot product kernel */
template <typename T>
__global__ void sequential_dot_product_kernel(const int n, T *a, T *b,
                                              T *result, Precision prec) {
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  /* compute */
  T sum = static_cast<T>(0.0);
  for (int i = 0; i < n; i++) {
    if (prec == Double) {
      sum = __dadd_rn(sum, __dmul_rn(a[i], b[i]));
    } else if (prec == Single) {
      sum = __fadd_rn(sum, __fmul_rn(a[i], b[i]));
    } else if (prec == Half) {
      sum = __hadd_rn(
          sum, __hmul_rn(static_cast<half>(a[i]), static_cast<half>(b[i])));
    } else {
      printf("<Cuda Error>: Invalid precision\n");
      return;
    }
  }
  /* copy */
  *result = sum;
}

__global__ void sequential_dot_product_model_kernel(
    const int n, double *a, double *b, double *result, Precision prec,
    BoundModel bound_model, const double beta_dist_alpha,
    const double beta_dist_beta, unsigned long long seed = 1234ULL) {
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  /* compute */
  double rounding_error[2];
  /* random state */
  curandState state;
  curand_init(seed, gid, 0, &state);
  /* compute */
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    /* sample rounding error delta */
    sample_rounding_error_distribution(2, rounding_error, prec, bound_model,
                                       beta_dist_alpha, beta_dist_beta, &state);
    /* get perturbation */
    rounding_error[0] = 1.0 + rounding_error[0];
    rounding_error[1] = 1.0 + rounding_error[1];
    /* compute */
    sum = __dmul_rn(
        __dadd_rn(sum, __dmul_rn(__dmul_rn(a[i], b[i]), rounding_error[0])),
        rounding_error[1]);
  }
  /* copy */
  *result = sum;
}

/* sequential dot product kernel launcher */
template <typename T>
void launch_sequential_dot_product_kernel(const int n,
                                          const std::vector<T> &h_a,
                                          const std::vector<T> &h_b,
                                          T *h_result, Precision prec,
                                          bool verbose) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initilaize */
  T *d_a, *d_b, *d_result;
  int size = n * sizeof(T);
  /* allocate memeory */
  cudaCheck(cudaMalloc((void **)&d_a, size));
  cudaCheck(cudaMalloc((void **)&d_b, size));
  cudaCheck(cudaMalloc((void **)&d_result, sizeof(T)));
  /* host to device */
  cudaCheck(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for sequential dot product in "
              << to_string(prec) << " precision" << std::endl;
  sequential_dot_product_kernel<<<gridDim, blockDim>>>(n, d_a, d_b, d_result,
                                                       prec);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  /* free */
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

void launch_sequential_dot_product_model_kernel(
    const int n, const std::vector<double> &h_a, const std::vector<double> &h_b,
    double *h_result, Precision prec, const gamma_config &gamma_cfg,
    bool verbose) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initilaize */
  double *d_a, *d_b, *d_result;
  int size = n * sizeof(double);
  BoundModel bound_model = gamma_cfg.bound_model;
  const double beta_dist_alpha = gamma_cfg.beta_dist_alpha;
  const double beta_dist_beta = gamma_cfg.beta_dist_beta;
  /* allocate memeory */
  cudaCheck(cudaMalloc((void **)&d_a, size));
  cudaCheck(cudaMalloc((void **)&d_b, size));
  cudaCheck(cudaMalloc((void **)&d_result, sizeof(double)));
  /* host to device */
  cudaCheck(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for sequential dot product model in "
              << to_string(Double) << " precision" << std::endl;
  sequential_dot_product_model_kernel<<<gridDim, blockDim>>>(
      n, d_a, d_b, d_result, prec, bound_model, beta_dist_alpha,
      beta_dist_beta);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(
      cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
  /* free */
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

/* initialize template */
template void launch_sequential_dot_product_kernel<double>(
    const int, const std::vector<double> &, const std::vector<double> &,
    double *, Precision, bool);
template void launch_sequential_dot_product_kernel<float>(
    const int, const std::vector<float> &, const std::vector<float> &, float *,
    Precision, bool);
template void launch_sequential_dot_product_kernel<half>(
    const int, const std::vector<half> &, const std::vector<half> &, half *,
    Precision, bool);
