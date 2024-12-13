#include <cuda_fp16.h>

#include <cassert>
#include <ctime>
#include <iostream>

#include "bounds.hpp"
#include "checks.cuh"
#include "config.hpp"
#include "convert.cuh"
#include "copy.cuh"
#include "definition.hpp"
#include "ebwd.hpp"
#include "matVecMult.cuh"
#include "sampler.cuh"
#include "utils.hpp"

template <typename T>
void __global__ RecursiveMatVecMultKernel(int N, T *a, T *x, T *result,
                                          Precision prec) {
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  if (gid < N) {
    T summation = static_cast<T>(0.0);
    for (int k = 0; k < N; k++) {
      if (prec == Double) {
        summation = __dadd_rn(summation, __dmul_rn(a[gid * N + k], x[k]));
      } else if (prec == Float) {
        summation = __fadd_rn(summation, __fmul_rn(a[gid * N + k], x[k]));
      } else if (prec == Half) {
        summation = __hadd_rn(
            summation, __hmul_rn(static_cast<half>(a[gid * N + k]), x[k]));
      } else {
        printf("invalid_argument: Invalid precision\n");
        return;
      }
    }
    result[gid] = summation;
  }
}

template <typename T>
void __global__ ModelRecursiveMatVecMultKernel(int N, T *a, T *x, T *result,
                                               Precision prec, double urd) {
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;

  double eta, xi;
  unsigned long long seed = 1234ULL;
  curandState state;
  curand_init(seed, gid, 0, &state);

  if (gid < N) {
    T summation = static_cast<T>(0.0);
    for (int k = 0; k < N; k++) {
      eta = 1.0 + sampleRelativeErrorKernel(urd, &state);
      xi = 1.0 + sampleRelativeErrorKernel(urd, &state);

      if (prec == Double) {
        summation = __dadd_rn(summation, __dmul_rn(a[gid * N + k], x[k]));
      } else if (prec == Float) {
        summation = __fadd_rn(summation, __fmul_rn(a[gid * N + k], x[k]));
      } else if (prec == Half) {
        summation = __hadd_rn(
            summation, __hmul_rn(static_cast<half>(a[gid * N + k]), x[k]));
      } else {
        printf("invalid_argument: Invalid precision\n");
        return;
      }
    }
    result[gid] = summation;
  }
}

void launchRecursiveMatVecMult(const int N, double *ebwd_float,
                               double *ebwd_half, double *ebwd_float_model,
                               double *ebwd_half_model,
                               unsigned long long seed) {
  // Declare
  double *a_double, *x_double, *result_double;
  double *a_double_abs, *x_double_abs, *result_double_abs;
  float *a_float, *x_float, *result_float;
  half *a_half, *x_half, *result_half;
  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  const int mat_size = N * N;

  Distribution dtype = config::distType;  // Distribution for sampling

  // Initialize
  cudaCheck(cudaMallocManaged(&a_double, mat_size * sizeof(double)));
  cudaCheck(cudaMallocManaged(&x_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double, N * sizeof(double)));

  cudaCheck(cudaMallocManaged(&a_double_abs, mat_size * sizeof(double)));
  cudaCheck(cudaMallocManaged(&x_double_abs, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double_abs, N * sizeof(double)));

  cudaCheck(cudaMallocManaged(&a_float, mat_size * sizeof(float)));
  cudaCheck(cudaMallocManaged(&x_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&result_float, N * sizeof(float)));

  cudaCheck(cudaMallocManaged(&a_half, mat_size * sizeof(half)));
  cudaCheck(cudaMallocManaged(&x_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&result_half, N * sizeof(half)));

  // Sample (in the lowest precision to avoid representation error)
  initializeVector(mat_size, a_half, dtype, seed);
  cudaCheck(cudaGetLastError());
  initializeVector(N, x_half, dtype, seed + 34577ULL);
  cudaCheck(cudaGetLastError());

  convertHalfToDouble(mat_size, a_half, a_double);
  cudaCheck(cudaGetLastError());
  convertHalfToDouble(N, x_half, x_double);
  cudaCheck(cudaGetLastError());

  convertHalfToFloat(mat_size, a_half, a_float);
  cudaCheck(cudaGetLastError());
  convertHalfToFloat(N, x_half, x_float);
  cudaCheck(cudaGetLastError());

  copyVector(mat_size, a_double, a_double_abs, true);
  cudaCheck(cudaGetLastError());
  copyVector(N, x_double, x_double_abs, true);
  cudaCheck(cudaGetLastError());

  // Compute the True matrix-vector product
  RecursiveMatVecMultKernel<<<gridDim, blockDim>>>(N, a_double, x_double,
                                                   result_double, Double);
  cudaCheck(cudaGetLastError());

  // Compute the Model matrix-vector product
  ModelRecursiveMatVecMultKernel<<<gridDim, blockDim>>>(N, a_double, x_double,
                                                        result_double, Double);

  // Compute the abs True dot product
  RecursiveMatVecMultKernel<<<gridDim, blockDim>>>(
      N, a_double_abs, x_double_abs, result_double_abs, Double);
  cudaCheck(cudaGetLastError());

  // Compute the Single precision matrix-vector product
  RecursiveMatVecMultKernel<<<gridDim, blockDim>>>(N, a_float, x_float,
                                                   result_float, Float);
  cudaCheck(cudaGetLastError());

  // Compute the Single precision matrix-vector product
  RecursiveMatVecMultKernel<<<gridDim, blockDim>>>(N, a_half, x_half,
                                                   result_half, Half);
  cudaCheck(cudaGetLastError());

  // Sanity check
  if (dtype == Ones) {
    cudaDeviceSynchronize();
    for (int ii = 0; ii < N; ii++) {
      assert(abs(result_double[ii] - N) < 1e-6 && "Invalid computation");
      assert(abs(result_double_abs[ii] - N) < 1e-6 && "Invalid computation");
      assert(abs(result_float[ii] - N) < 1e-6 && "Invalid computation");
      assert(abs(static_cast<double>(result_half[ii]) - N) < 1e-6 &&
             "Invalid computation");
    }
  }

  // Synchronize
  cudaDeviceSynchronize();

  // Compute the backward error
  computeBackwardErrorMatVecMult(N, result_double, result_float,
                                 result_double_abs, ebwd_float);

  computeBackwardErrorDotProduct(N, result_double, result_half,
                                 result_double_abs, ebwd_half);

  std::cout << *ebwd_float << std::endl;
}

void launchMatVecMultExperiment(int N_lower, int bit_shift, int max_shift,
                                int num_exps, double confidence) {
  // Square matrix-vector product
  int N = N_lower;
  const int width_int = 6;      // For I/O
  const int width_double = 15;  // For I/O

  double *ebwd, *ebwd_model;
  double *ebwd_bound_det, *ebwd_bound_hoeff, *ebwd_bound_bern;
  ebwd_bound_det =
      static_cast<double *>(malloc(2 * max_shift * sizeof(double)));
  ebwd_bound_hoeff =
      static_cast<double *>(malloc(2 * max_shift * sizeof(double)));
  ebwd_bound_bern =
      static_cast<double *>(malloc(2 * max_shift * sizeof(double)));
  ebwd =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));
  ebwd_model =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));

  for (int ii = 0; ii < 1; ii++) {
    std::cout << "Problem size: " << N << std::endl;

    // Compute backward bound (Float)
    ebwd_bound_det[ii] =
        matVecBackwardBound(N, Float, Deterministic, confidence);
    ebwd_bound_hoeff[ii] = matVecBackwardBound(N, Float, Hoeffding, confidence);
    ebwd_bound_bern[ii] = matVecBackwardBound(N, Float, Bernstein, confidence);

    // Compute backward bound (Half)
    ebwd_bound_det[max_shift + ii] =
        matVecBackwardBound(N, Half, Deterministic, confidence);
    ebwd_bound_hoeff[max_shift + ii] =
        matVecBackwardBound(N, Half, Hoeffding, confidence);
    ebwd_bound_bern[max_shift + ii] =
        matVecBackwardBound(N, Half, Bernstein, confidence);

    /* printf("Det: %.5e Higham: %.5e Bern: %.5e\n", ebwd_bound_det[ii], */
    /*        ebwd_bound_hoeff[ii], ebwd_bound_bern[ii]); */

    // Carry experiment of matrix-vector products
    for (int jj = 0; jj < num_exps; jj++) {
      // Experiment seed
      unsigned long long base_seed =
          static_cast<unsigned long long>(std::time(nullptr));
      base_seed += sqrt(jj * 354);  // Experimental seed
      base_seed += sqrt(ii * 231);  // Problem size seed

      // Single experiment
      launchRecursiveMatVecMult(
          N, &ebwd[ii * num_exps + jj],
          &ebwd[max_shift * num_exps + ii * num_exps + jj],
          &ebwd_model[ii * num_exps + jj],
          &ebwd_model[max_shift * num_exps + ii * num_exps + jj], base_seed);

      // Single sequential experiment
    }

    // Increase the Matrix/vector size
    N = N << bit_shift;
  }
}
