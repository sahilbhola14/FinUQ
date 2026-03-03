#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <cassert>
#include <iostream>

#include "definition.hpp"
#include "dot_product_cuda.cuh"
#include "rounding_error_model.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* (sequential) dot product kernel */
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

/* (block) dot product kernel */
template <typename T>
__global__ void block_dot_product_kernel(const int n, T *a, T *b, T *partial,
                                         Precision prec) {
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  // Dynamic shared memory for per-block reduction.
  extern __shared__ unsigned char sdata_raw[];
  T *sdata = reinterpret_cast<T *>(sdata_raw);

  // Per-thread sum
  T sum = static_cast<T>(0.0);
  if (gid < n) {
    if (prec == Double) {
      sum = __dmul_rn(a[gid], b[gid]);
    } else if (prec == Single) {
      sum = __fmul_rn(a[gid], b[gid]);
    } else if (prec == Half) {
      sum = __hmul_rn(static_cast<half>(a[gid]), static_cast<half>(b[gid]));
    } else {
      printf("<Cuda Error>: Invalid precision\n");
      return;
    }
  }

  sdata[tid] = sum;
  __syncthreads();

  // Tree reduction
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (prec == Double) {
        sdata[tid] = __dadd_rn(sdata[tid], sdata[tid + stride]);
      } else if (prec == Single) {
        sdata[tid] = __fadd_rn(sdata[tid], sdata[tid + stride]);
      } else {
        sdata[tid] = __hadd_rn(static_cast<half>(sdata[tid]),
                               static_cast<half>(sdata[tid + stride]));
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = sdata[0];
  }
}

__global__ void sequential_dot_product_model_kernel(
    const int n, double *a, double *b, double *result, Precision prec,
    BoundModel bound_model, const double beta_dist_alpha,
    const double beta_dist_beta, const int experiment_id,
    unsigned long long seed = 1234ULL) {
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  /* compute */
  double rounding_error[2];
  /* random state */
  curandState state;
  curand_init(seed, experiment_id * gridDim.x * blockDim.x + gid, 0, &state);
  /* compute */
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    /* sample rounding error delta */
    sample_rounding_error_distribution(2, rounding_error, prec, bound_model,
                                       beta_dist_alpha, beta_dist_beta, &state);
    /* printf("%.5e , %.5e\n", rounding_error[0], rounding_error[1]); */
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

/* (sequetnial) dot product kernel launcher */
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
  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_result));
}

/*  (block) dot product kernel launcher */
template <typename T>
void launch_block_dot_product_kernel(const int n, const std::vector<T> &h_a,
                                     const std::vector<T> &h_b, T *h_result,
                                     Precision prec, const int block_dim,
                                     bool verbose) {
  assert(block_dim > 0 && "block_dim must be positive");
  assert((block_dim & (block_dim - 1)) == 0 &&
         "block_dim must be a power of 2 for tree reduction");

  /* kernel parameters */
  dim3 blockDim = block_dim;
  dim3 gridDim = get_grid_dim(n, block_dim);
  /* initilaize */
  T *d_a, *d_b, *d_result, *d_partial;
  int size = n * sizeof(T);
  /* allocate memeory */
  cudaCheck(cudaMalloc((void **)&d_a, size));
  cudaCheck(cudaMalloc((void **)&d_b, size));
  cudaCheck(cudaMalloc((void **)&d_partial, gridDim.x * sizeof(T)));
  cudaCheck(cudaMalloc((void **)&d_result, sizeof(T)));
  /* host to device */
  cudaCheck(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for block dot product in " << to_string(prec)
              << " precision with " << gridDim.x << "grids" << std::endl;
  block_dot_product_kernel<<<gridDim, blockDim, blockDim.x * sizeof(T)>>>(
      n, d_a, d_b, d_partial, prec);
  cudaCheck(cudaGetLastError());
  // /* device to host */
  // cudaCheck(cudaMemcpy(h_result, d_result, sizeof(T),
  // cudaMemcpyDeviceToHost));
  // /* free */
  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_partial));
  cudaCheck(cudaFree(d_result));
}

void launch_sequential_dot_product_model_kernel(
    const int n, const std::vector<double> &h_a, const std::vector<double> &h_b,
    double *h_result, Precision prec, const gamma_config &gamma_cfg,
    const int experiment_id, bool verbose) {
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
      n, d_a, d_b, d_result, prec, bound_model, beta_dist_alpha, beta_dist_beta,
      experiment_id);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(
      cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
  /* free */
  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_result));
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

template void launch_block_dot_product_kernel<double>(
    const int, const std::vector<double> &, const std::vector<double> &,
    double *, Precision, const int, bool);
template void launch_block_dot_product_kernel<float>(const int,
                                                     const std::vector<float> &,
                                                     const std::vector<float> &,
                                                     float *, Precision,
                                                     const int, bool);
template void launch_block_dot_product_kernel<half>(const int,
                                                    const std::vector<half> &,
                                                    const std::vector<half> &,
                                                    half *, Precision,
                                                    const int, bool);
