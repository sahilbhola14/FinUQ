#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cassert>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "addition.cuh"
#include "bounds.hpp"
#include "checks.cuh"
#include "config.hpp"
#include "convert.cuh"
#include "copy.cuh"
#include "definition.hpp"
#include "dotProduct.cuh"
#include "ebwd.hpp"
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

template <typename T>
__global__ void RecursiveDotProductKernel(int N, T *x, T *y, T *result,
                                          Precision prec) {
  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  T summation = static_cast<T>(0.0);

  for (int i = 0; i < N; i++) {
    if (prec == Double) {
      summation = __dadd_rn(summation, __dmul_rn(x[i], y[i]));
    } else if (prec == Float) {
      summation = __fadd_rn(summation, __fmul_rn(x[i], y[i]));
    } else if (prec == Half) {
      summation =
          __hadd_rn(summation, __hmul_rn(static_cast<half>(x[i]), y[i]));
    } else {
      printf("invalid_argument: Invalid precision\n");
      return;
    }
  }

  *result = summation;
}

__global__ void ModelRecursiveDotProductKernel(int N, double *x, double *y,
                                               double *result, double urd) {
  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  double summation = 0.0;
  double eta, xi;
  unsigned long long seed = 1234ULL;
  curandState state;
  curand_init(seed, gid, 0, &state);

  for (int i = 0; i < N; i++) {
    eta = 1.0 + sampleRelativeErrorKernel(urd, &state);
    xi = 1.0 + sampleRelativeErrorKernel(urd, &state);
    // Model: s_i = (s_{i-1} +  a_i b_i (1 + eta_i))(1 + xi_i)
    summation = __dmul_rn(
        __dadd_rn(summation, __dmul_rn(__dmul_rn(x[i], y[i]), eta)), xi);
  }

  *result = summation;
}

void launchcublasDDot(int N, double *x, double *y, double *result) {
  // Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Compute
  cublasDdot(handle, N, x, 1, y, 1, result);

  /* // Destroy handle */
  cublasDestroy(handle);
}

void launchRecursiveDotProduct(const int N, double *ebwd_float,
                               double *ebwd_half, double *ebwd_float_model,
                               double *ebwd_half_model,
                               unsigned long long seed) {
  // Note, blockDim and gridDim should be 1 for accumulation of error
  // This is the recursive summation
  double *x_double, *y_double, *result_double;
  double *result_double_model, *result_double_model_abs;
  double *x_double_abs, *y_double_abs, *result_double_abs;
  float *x_float, *y_float, *result_float;
  half *x_half, *y_half, *result_half;
  double urd_float = computeUnitRoundOff(Float);
  double urd_half = computeUnitRoundOff(Half);

  Distribution dtype = config::distType;  // Distribution for sampling

  dim3 blockDim = 1;
  dim3 gridDim = 1;

  // Allocate memory
  cudaCheck(cudaMallocManaged(&x_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double, sizeof(double)));

  cudaCheck(cudaMallocManaged(&result_double_model, sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double_model_abs, sizeof(double)));

  cudaCheck(cudaMallocManaged(&x_double_abs, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double_abs, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double_abs, sizeof(double)));

  cudaCheck(cudaMallocManaged(&x_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&y_float, N * sizeof(float)));
  cudaCheck(cudaMallocManaged(&result_float, sizeof(float)));

  cudaCheck(cudaMallocManaged(&x_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&y_half, N * sizeof(half)));
  cudaCheck(cudaMallocManaged(&result_half, sizeof(half)));

  // Sample (in the lowest precision to avoid representation error)
  initializeVector(N, x_half, dtype, seed);
  initializeVector(N, y_half, dtype, seed + 34577ULL);

  convertHalfToDouble(N, x_half, x_double);
  convertHalfToDouble(N, y_half, y_double);

  convertHalfToFloat(N, x_half, x_float);
  convertHalfToFloat(N, y_half, y_float);

  copyVector(N, x_double, x_double_abs, true);
  copyVector(N, y_double, y_double_abs, true);

  // Compute the True dot product
  launchcublasDDot(N, x_double, y_double, result_double);

  // Compute the Model dot product
  ModelRecursiveDotProductKernel<<<gridDim, blockDim>>>(
      N, x_double, y_double, result_double_model, urd_float);
  cudaCheck(cudaGetLastError());

  // Compute the abs True dot product
  launchcublasDDot(N, x_double_abs, y_double_abs, result_double_abs);

  // Compute the abs Model dot product
  ModelRecursiveDotProductKernel<<<gridDim, blockDim>>>(
      N, x_double_abs, y_double_abs, result_double_model_abs, urd_float);

  // Compute Single precision dot product
  RecursiveDotProductKernel<<<gridDim, blockDim>>>(N, x_float, y_float,
                                                   result_float, Float);
  cudaCheck(cudaGetLastError());

  // Compute Half precision dot product
  RecursiveDotProductKernel<<<gridDim, blockDim>>>(N, x_half, y_half,
                                                   result_half, Half);
  cudaCheck(cudaGetLastError());

  // Synchronize
  cudaDeviceSynchronize();

  // Compute the backward error
  computeBackwardErrorDotProduct(result_double, result_float, result_double_abs,
                                 ebwd_float);
  computeBackwardErrorDotProduct(result_double, result_half, result_double_abs,
                                 ebwd_half);

  computeBackwardErrorDotProduct(result_double_model, result_float,
                                 result_double_abs, ebwd_float_model);
  computeBackwardErrorDotProduct(result_double_model, result_half,
                                 result_double_abs, ebwd_half_model);

  // Free memory
  cudaFree(x_double);
  cudaFree(y_double);
  cudaFree(result_double);

  cudaFree(result_double_model);
  cudaFree(result_double_model_abs);

  cudaFree(x_double_abs);
  cudaFree(y_double_abs);
  cudaFree(result_double_abs);

  cudaFree(x_float);
  cudaFree(y_float);
  cudaFree(result_float);

  cudaFree(x_half);
  cudaFree(y_half);
  cudaFree(result_half);
}

void launchDotProductSingleRun(const int N, double *ebwd_float,
                               double *ebwd_half, unsigned long long seed) {
  // Note: Interstingly, the parallel operaitons results in less accumulation of
  // error.
  double *x_double, *y_double, *partial_double, *result_double;
  double *result_cublas, *result_cublas_abs;
  double *x_double_abs, *y_double_abs, *partial_double_abs, *result_double_abs;
  float *x_float, *y_float, *partial_float, *result_float;
  half *x_half, *y_half, *partial_half, *result_half;
  int cacheSize;

  Distribution dtype = config::distType;  // Distribution for sampling

  dim3 blockDim = config::blockSize;
  dim3 gridDim = getGridSize(blockDim.x, N);
  /* int cacheSize; */

  /* printf("----- config -----\n"); */
  /* printf("Vector size: %d\n", N); */
  /* printf("blockDim: %d gridDim: %d\n", blockDim.x, gridDim.x); */
  /* printf("------------------\n"); */

  // Allocate memory
  cudaCheck(cudaMallocManaged(&x_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&partial_double, gridDim.x * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double, sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_cublas, sizeof(double)));

  cudaCheck(cudaMallocManaged(&x_double_abs, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double_abs, N * sizeof(double)));
  cudaCheck(cudaMallocManaged(&partial_double_abs, gridDim.x * sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_double_abs, sizeof(double)));
  cudaCheck(cudaMallocManaged(&result_cublas_abs, sizeof(double)));

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

  copyVector(N, x_double, x_double_abs, true);
  copyVector(N, y_double, y_double_abs, true);

  // Double precision Dot product
  cacheSize = blockDim.x * sizeof(double);
  dotProductKernel<<<gridDim, blockDim, cacheSize>>>(N, x_double, y_double,
                                                     partial_double, Double);
  cudaCheck(cudaGetLastError());
  allSumReduce(gridDim.x, partial_double, result_double, Double);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();

  // cuBlas Dot
  launchcublasDDot(N, x_double, y_double, result_cublas);

  // Double precision Dot product of absolute values
  cacheSize = blockDim.x * sizeof(double);
  dotProductKernel<<<gridDim, blockDim, cacheSize>>>(
      N, x_double_abs, y_double_abs, partial_double_abs, Double);
  cudaCheck(cudaGetLastError());
  allSumReduce(gridDim.x, partial_double_abs, result_double_abs, Double);
  cudaCheck(cudaGetLastError());
  cudaDeviceSynchronize();

  // cublas Dot
  launchcublasDDot(N, x_double_abs, y_double_abs, result_cublas_abs);

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
  /* std::cout << "Result: " << std::scientific << std::setprecision(15) <<
   * *result_double << ", " */
  /*           << *result_float << ", " << static_cast<double>(*result_half) */
  /*           << std::endl; */

  // Compute the backward error
  /* computeBackwardErrorDotProduct(result_double, result_float,
   * result_double_abs, ebwd_float); */
  computeBackwardErrorDotProduct(result_cublas, result_float, result_cublas_abs,
                                 ebwd_float);
  computeBackwardErrorDotProduct(result_cublas, result_half, result_cublas_abs,
                                 ebwd_half);
  /* std::cout << *ebwd_float << ", " << *ebwd_half << std::endl; */

  // Free memory
  cudaFree(x_double);
  cudaFree(y_double);
  cudaFree(partial_double);
  cudaFree(result_double);

  cudaFree(x_double_abs);
  cudaFree(y_double_abs);
  cudaFree(partial_double_abs);
  cudaFree(result_double_abs);

  cudaFree(x_float);
  cudaFree(y_float);
  cudaFree(partial_float);
  cudaFree(result_float);

  cudaFree(x_half);
  cudaFree(y_half);
  cudaFree(partial_half);
  cudaFree(result_half);
}

void launchDotProductExperiment(int N_lower, int bit_shift, int max_shift,
                                int num_exps, double confidence) {
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

  for (int ii = 0; ii < max_shift; ii++) {
    std::cout << "Problem size: " << N << std::endl;

    // Compute backward bound (Float)
    ebwd_bound_det[ii] =
        dotProductBackwardBound(N, Float, Deterministic, confidence);
    ebwd_bound_hoeff[ii] =
        dotProductBackwardBound(N, Float, Hoeffding, confidence);
    ebwd_bound_bern[ii] =
        dotProductBackwardBound(N, Float, Bernstein, confidence);

    // Compute backward bound (Half)
    ebwd_bound_det[max_shift + ii] =
        dotProductBackwardBound(N, Half, Deterministic, confidence);
    ebwd_bound_hoeff[max_shift + ii] =
        dotProductBackwardBound(N, Half, Hoeffding, confidence);
    ebwd_bound_bern[max_shift + ii] =
        dotProductBackwardBound(N, Half, Bernstein, confidence);

    // Carry experiment of dot products
    for (int jj = 0; jj < num_exps; jj++) {
      // Experiment seed
      unsigned long long base_seed =
          static_cast<unsigned long long>(std::time(nullptr));
      base_seed += sqrt(jj * 354);  // Experimental seed
      base_seed += sqrt(ii * 231);  // Problem size seed

      // Single parallel experiment
      /* launchDotProductSingleRun(N, &ebwd[ii*num_exps + jj],
       * &ebwd[max_shift*num_exps + ii*num_exps + jj], base_seed); */
      // Single sequential experiment
      launchRecursiveDotProduct(
          N, &ebwd[ii * num_exps + jj],
          &ebwd[max_shift * num_exps + ii * num_exps + jj],
          &ebwd_model[ii * num_exps + jj],
          &ebwd_model[max_shift * num_exps + ii * num_exps + jj], base_seed);
    }

    N = N << bit_shift;
  }

  /* // Store the data // */
  std::ofstream outfile;
  std::string filename;

  // Store ebwd for FP32
  filename = "dotProduct_ebwd_fp32.txt";
  outfile.open(filename);

  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward error for dot product computed in FP32 "
             "arithmetic (confidence: "
          << confidence << ")" << std::endl;

  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;

  outfile << std::setw(width_int) << "Deter: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_det[ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Hoeff: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_hoeff[ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Berns: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_bern[ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int + 1) << "    ";
  for (int ii = 0; ii < num_exps; ii++) {
    for (int jj = 0; jj < max_shift; jj++) {
      outfile << std::setw(width_double) << std::scientific
              << std::setprecision(8) << ebwd[jj * num_exps + ii];
      if (jj < max_shift - 1) outfile << ", ";
    }
    outfile << std::endl;
    outfile << std::setw(width_int + 1) << "    ";
  }

  outfile.close();

  // Store ebwd for FP16
  filename = "dotProduct_ebwd_fp16.txt";
  outfile.open(filename);

  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward error for dot product computed in FP16 "
             "arithmetic (confidence: "
          << confidence << ")" << std::endl;

  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;

  outfile << std::setw(width_int) << "Deter: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_det[max_shift + ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Hoeff: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_hoeff[max_shift + ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Berns: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_bern[max_shift + ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int + 1) << "    ";
  for (int ii = 0; ii < num_exps; ii++) {
    for (int jj = 0; jj < max_shift; jj++) {
      outfile << std::setw(width_double) << std::scientific
              << std::setprecision(8)
              << ebwd[max_shift * num_exps + jj * num_exps + ii];
      if (jj < max_shift - 1) outfile << ", ";
    }
    outfile << std::endl;
    outfile << std::setw(width_int + 1) << "    ";
  }

  outfile.close();

  // Store ebwd computed with model for FP32
  filename = "dotProduct_ebwd_model_fp32.txt";
  outfile.open(filename);

  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward error (computed with Model) for dot "
             "product computed in FP32 arithmetic (confidence: "
          << confidence << ")" << std::endl;

  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;

  outfile << std::setw(width_int) << "Deter: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_det[ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Hoeff: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_hoeff[ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Berns: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_bern[ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int + 1) << "    ";
  for (int ii = 0; ii < num_exps; ii++) {
    for (int jj = 0; jj < max_shift; jj++) {
      outfile << std::setw(width_double) << std::scientific
              << std::setprecision(8) << ebwd_model[jj * num_exps + ii];
      if (jj < max_shift - 1) outfile << ", ";
    }
    outfile << std::endl;
    outfile << std::setw(width_int + 1) << "    ";
  }

  outfile.close();

  // Store ebwd computed with Model for FP16
  filename = "dotProduct_ebwd_model_fp16.txt";
  outfile.open(filename);

  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward error for dot product computed in FP16 "
             "arithmetic (confidence: "
          << confidence << ")" << std::endl;

  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;

  outfile << std::setw(width_int) << "Deter: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_det[max_shift + ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Hoeff: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_hoeff[max_shift + ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int) << "Berns: ";
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << std::scientific
            << std::setprecision(8) << ebwd_bound_bern[max_shift + ii];
    if (ii < max_shift - 1) outfile << ", ";
  }
  outfile << std::endl;
  outfile << std::setw(width_int + 1) << "    ";
  for (int ii = 0; ii < num_exps; ii++) {
    for (int jj = 0; jj < max_shift; jj++) {
      outfile << std::setw(width_double) << std::scientific
              << std::setprecision(8)
              << ebwd_model[max_shift * num_exps + jj * num_exps + ii];
      if (jj < max_shift - 1) outfile << ", ";
    }
    outfile << std::endl;
    outfile << std::setw(width_int + 1) << "    ";
  }

  // Free the memory
  free(ebwd);
  free(ebwd_model);
  free(ebwd_bound_det);
  free(ebwd_bound_hoeff);
  free(ebwd_bound_bern);
}
