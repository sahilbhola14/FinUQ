#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <cassert>
#include <iostream>

#include "definition.hpp"
#include "dot_product_cuda.cuh"
#include "rounding_error_model.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

// data processed per thread block
constexpr int BLOCK_SIZE = 256;

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
                                         Precision prec, const int tile_size) {
  // parameters
  int tid = threadIdx.x;
  int gid = (threadIdx.x * tile_size) + blockIdx.x * BLOCK_SIZE;
  // dynamic memory allocation for per-block reduction
  extern __shared__ unsigned char sdata_raw[];
  T *sdata = reinterpret_cast<T *>(sdata_raw);

  // Per-thread tiled accumulation.
  T sum = static_cast<T>(0.0);
  for (int j = 0; j < tile_size; j++) {
    int idx = gid + j;
    if (idx < n) {
      if (prec == Double) {
        sum = __dadd_rn(sum, __dmul_rn(a[idx], b[idx]));
      } else if (prec == Single) {
        sum = __fadd_rn(sum, __fmul_rn(a[idx], b[idx]));
      } else if (prec == Half) {
        sum = __hadd_rn(sum, __hmul_rn(static_cast<half>(a[idx]),
                                       static_cast<half>(b[idx])));
      } else {
        printf("<Cuda Error>: Invalid precision\n");
        return;
      }
    }
  }

  sdata[tid] = sum;
  __syncthreads();

  // Tree reduction within the block.
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

/* partial reduction kernel */
template <typename T>
__global__ void reduce_partial_sum_kernel(const int n, const T *in, T *out,
                                          Precision prec) {
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ unsigned char sdata_raw[];
  T *sdata = reinterpret_cast<T *>(sdata_raw);

  T sum = static_cast<T>(0.0);
  if (gid < n) {
    sum = in[gid];
  }

  sdata[tid] = sum;
  __syncthreads();

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
    out[blockIdx.x] = sdata[0];
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
                                     Precision prec, const int tile_size,
                                     bool verbose) {
  /* kernel parameters */
  //
  assert(tile_size > 0 && "tile_size must be positive");
  assert((tile_size & (tile_size - 1)) == 0 &&
         "tile_size must be a power of 2 for tree reduction");
  assert((tile_size <= BLOCK_SIZE) && (BLOCK_SIZE % tile_size) == 0 &&
         "tile_size must less that BLOCK_SIZE and perfectly divide BLOCK_SIZE");

  dim3 blockDim = BLOCK_SIZE / tile_size;      // number of threads per block
  dim3 gridDim = get_grid_dim(n, BLOCK_SIZE);  // number of blocks

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
              << " precision with tile_size" << tile_size << "and " << gridDim.x
              << "grids" << std::endl;
  block_dot_product_kernel<<<gridDim, blockDim, blockDim.x * sizeof(T)>>>(
      n, d_a, d_b, d_partial, prec, tile_size);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  // Reduce partial sums to one scalar, possibly across multiple passes.
  int partial_size = static_cast<int>(gridDim.x);
  T *d_curr = d_partial;
  T *d_next = nullptr;

  while (partial_size > 1) {
    int reduce_grid = get_grid_dim(partial_size, blockDim.x);
    cudaCheck(cudaMalloc((void **)&d_next, reduce_grid * sizeof(T)));
    reduce_partial_sum_kernel<<<reduce_grid, blockDim,
                                blockDim.x * sizeof(T)>>>(partial_size, d_curr,
                                                          d_next, prec);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    if (d_curr != d_partial) {
      cudaCheck(cudaFree(d_curr));
    }
    d_curr = d_next;
    d_next = nullptr;
    partial_size = reduce_grid;
  }

  cudaCheck(cudaMemcpy(d_result, d_curr, sizeof(T), cudaMemcpyDeviceToDevice));
  cudaCheck(cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

  if (d_curr != d_partial) {
    cudaCheck(cudaFree(d_curr));
  }

  /* free */
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
