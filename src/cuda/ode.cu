#include <assert.h>
#include <cuda_fp16.h>

#include <ctime>
#include <stdexcept>

#include "bounds.hpp"
#include "checks.cuh"
#include "config.hpp"
#include "convert.cuh"
#include "definition.hpp"
#include "ebwd.hpp"
#include "ode.cuh"
#include "sampler.cuh"
#include "utils.hpp"

__global__ void getDiagonalsKernel(const int N, half *theta_1, half *theta_2,
                                   half *sub_diag, half *main_diag,
                                   half *super_diag) {
  // Declaration
  const int N_inner = N - 1;
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  half dx = static_cast<half>(1.0 / N);
  half dx_sq = __hmul_rn(dx, dx);
  half one_half = static_cast<half>(1.0 / 2.0);
  half one = static_cast<half>(1.0);

  half half_dx = __hmul_rn(one_half, dx);

  if (gid < N_inner) {
    half x = __hmul_rn(__int2half_rn(gid + 1), dx);
    half x_plus = __hadd_rn(x, dx);
    sub_diag[gid] = __hadd_rn(one, __hmul_rn(*theta_1, __hsub_rn(x, half_dx)));
    super_diag[gid] =
        __hadd_rn(one, __hmul_rn(*theta_1, __hsub_rn(x_plus, half_dx)));
    main_diag[gid] = __hneg(__hadd_rn(sub_diag[gid], super_diag[gid]));
  }
}

__global__ void getRhsKernel(const int N, half *theta_1, half *theta_2,
                             half *rhs) {
  // Declaration
  const int N_inner = N - 1;
  int tid = threadIdx.x;
  int gid = tid + blockIdx.x * blockDim.x;
  half dx = static_cast<half>(1.0 / N);
  half dx_sq = __hmul_rn(dx, dx);
  half theta_2_sq = __hmul_rn(*theta_2, *theta_2);
  half fifty = static_cast<half>(50.0);
  if (gid < N_inner) {
    rhs[gid] = __hneg(__hmul_rn(__hmul_rn(fifty, dx_sq), theta_2_sq));
  }
}

template <typename T>
__global__ void getDecompositionKernel(const int N, half *sub_diag,
                                       half *main_diag, half *super_diag, T *a,
                                       T *b, Precision prec) {
  const int N_inner = N - 1;

  if (prec == Double) {
    b[0] = static_cast<double>(main_diag[0]);
  } else if (prec == Float) {
    b[0] = static_cast<float>(main_diag[0]);
  } else if (prec == Half) {
    b[0] = main_diag[0];
  } else {
    printf("Invalid argument\n");
    return;
  }

  for (int ii = 1; ii < N_inner; ii++) {
    if (prec == Double) {
      a[ii] = __ddiv_rn(static_cast<double>(sub_diag[ii]), b[ii - 1]);
      b[ii] =
          __dsub_rn(static_cast<double>(main_diag[ii]),
                    __dmul_rn(a[ii], static_cast<double>(super_diag[ii - 1])));
    } else if (prec == Float) {
      a[ii] = __fdiv_rn(static_cast<float>(sub_diag[ii]), b[ii - 1]);
      b[ii] =
          __fsub_rn(static_cast<float>(main_diag[ii]),
                    __fmul_rn(a[ii], static_cast<float>(super_diag[ii - 1])));
    } else if (prec == Half) {
      a[ii] = __hdiv(sub_diag[ii], b[ii - 1]);
      b[ii] = __hsub_rn(main_diag[ii], __hmul_rn(a[ii], super_diag[ii - 1]));
    } else {
      printf("Invalid argument\n");
      return;
    }
  }
}

template <typename T>
__global__ void forwardSubstitutionKernel(const int N, T *a, T *y, half *rhs,
                                          Precision prec) {
  const int N_inner = N - 1;
  if (prec == Double) {
    y[0] = static_cast<double>(rhs[0]);
  } else if (prec == Float) {
    y[0] = static_cast<float>(rhs[0]);
  } else if (prec == Half) {
    y[0] = rhs[0];
  } else {
    printf("Invalid argument\n");
    return;
  }

  for (int ii = 1; ii < N_inner; ii++) {
    if (prec == Double) {
      y[ii] =
          __dsub_rn(static_cast<double>(rhs[ii]), __dmul_rn(a[ii], y[ii - 1]));
    } else if (prec == Float) {
      y[ii] =
          __fsub_rn(static_cast<float>(rhs[ii]), __fmul_rn(a[ii], y[ii - 1]));
    } else if (prec == Half) {
      y[ii] = __hsub_rn(rhs[ii], __hmul_rn(a[ii], y[ii - 1]));
    } else {
      printf("Invalid argument\n");
      return;
    }
  }
}

template <typename T>
__global__ void backwardSubstitutionKernel(const int N, T *b, T *y, T *u,
                                           half *super_diag, Precision prec) {
  const int N_inner = N - 1;
  if (prec == Double) {
    u[N_inner - 1] = __ddiv_rn(y[N_inner - 1], b[N_inner - 1]);
  } else if (prec == Float) {
    u[N_inner - 1] = __fdiv_rn(y[N_inner - 1], b[N_inner - 1]);
  } else if (prec == Half) {
    u[N_inner - 1] = __hdiv(y[N_inner - 1], b[N_inner - 1]);
  } else {
    printf("Invalid argument\n");
    return;
  }

  for (int ii = N_inner - 2; ii >= 0; ii--) {
    if (prec == Double) {
      u[ii] = __ddiv_rn(
          __dsub_rn(y[ii],
                    __dmul_rn(static_cast<double>(super_diag[ii]), u[ii + 1])),
          b[ii]);
    } else if (prec == Float) {
      u[ii] = __fdiv_rn(
          __fsub_rn(y[ii],
                    __fmul_rn(static_cast<float>(super_diag[ii]), u[ii + 1])),
          b[ii]);
    } else if (prec == Half) {
      u[ii] =
          __hdiv(__hsub_rn(y[ii], __hmul_rn(super_diag[ii], u[ii + 1])), b[ii]);
    } else {
      printf("Invalid argument\n");
      return;
    }
  }
}

template <typename T>
__global__ void reimannIntegrationKernel(const int N, T *u, T *p,
                                         Precision prec) {
  const int N_inner = N - 1;
  const T dx = static_cast<T>(
      static_cast<half>(1.0 / N));  // Assumes no representation error
  T summation = static_cast<T>(0.0);

  for (int ii = 0; ii < N_inner; ii++) {
    if (prec == Double) {
      summation = __dadd_rn(summation, u[ii]);
    } else if (prec == Float) {
      summation = __fadd_rn(summation, u[ii]);
    } else if (prec == Half) {
      summation = __hadd_rn(summation, u[ii]);
    } else {
      printf("Invalid argument\n");
      return;
    }
  }

  if (prec == Double) {
    summation = __dmul_rn(summation, dx);
  } else if (prec == Float) {
    summation = __fmul_rn(summation, dx);
  } else if (prec == Half) {
    summation = __hmul_rn(summation, dx);
  } else {
    printf("Invalid argument\n");
    return;
  }

  *p = summation;
}

void launchODE(const int N, half *theta_1, half *theta_2, double *ebwd_float,
               double *ebwd_half, double *ebwd_float_model,
               double *ebwd_half_model, unsigned long long seed) {
  double dx = 1.0 / N;  // gridSize

  half *sub_diag, *main_diag, *super_diag,
      *rhs;  // Always in lower precision to avoid represenation error

  double *sub_diag_double, *main_diag_double, *super_diag_double, *rhs_double;

  double *a_double, *b_double, *y_double, *u_double, *p_double;
  float *a_float, *b_float, *y_float, *u_float, *p_float;
  half *a_half, *b_half, *y_half, *u_half, *p_half;

  const int N_inner = N - 1;

  dim3 blockDim = config::blockSize;
  dim3 gridDim =
      getGridSize(blockDim.x, N_inner);  // Map inner elements to each thread

  // Allocation

  cudaCheck(cudaMallocManaged(&sub_diag, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&main_diag, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&super_diag, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&rhs, N_inner * sizeof(half)));

  cudaCheck(cudaMallocManaged(&sub_diag_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&main_diag_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&super_diag_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&rhs_double, N_inner * sizeof(double)));

  cudaCheck(cudaMallocManaged(&a_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&b_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&u_double, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&p_double, sizeof(double)));

  cudaCheck(cudaMallocManaged(&a_float, N_inner * sizeof(float)));
  cudaCheck(cudaMallocManaged(&b_float, N_inner * sizeof(float)));
  cudaCheck(cudaMallocManaged(&y_float, N_inner * sizeof(float)));
  cudaCheck(cudaMallocManaged(&u_float, N_inner * sizeof(float)));
  cudaCheck(cudaMallocManaged(&p_float, sizeof(float)));

  cudaCheck(cudaMallocManaged(&a_half, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&b_half, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&y_half, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&u_half, N_inner * sizeof(half)));
  cudaCheck(cudaMallocManaged(&p_half, sizeof(half)));

  // Compute the Diagonals in lower precision
  getDiagonalsKernel<<<gridDim, blockDim>>>(N, theta_1, theta_2, sub_diag,
                                            main_diag, super_diag);
  cudaCheck(cudaGetLastError());

  convertHalfToDouble(N_inner, sub_diag, sub_diag_double);
  convertHalfToDouble(N_inner, main_diag, main_diag_double);
  convertHalfToDouble(N_inner, super_diag, super_diag_double);

  // Comute the rhs
  getRhsKernel<<<gridDim, blockDim>>>(N, theta_1, theta_2, rhs);
  cudaCheck(cudaGetLastError());

  convertHalfToDouble(N_inner, rhs, rhs_double);
  cudaDeviceSynchronize();

  // Compute the decomposition
  getDecompositionKernel<<<1, 1>>>(N, sub_diag, main_diag, super_diag, a_double,
                                   b_double, Double);
  cudaCheck(cudaGetLastError());

  getDecompositionKernel<<<1, 1>>>(N, sub_diag, main_diag, super_diag, a_float,
                                   b_float, Float);
  cudaCheck(cudaGetLastError());

  getDecompositionKernel<<<1, 1>>>(N, sub_diag, main_diag, super_diag, a_half,
                                   b_half, Half);
  cudaCheck(cudaGetLastError());

  // Forward Substitution
  forwardSubstitutionKernel<<<1, 1>>>(N, a_double, y_double, rhs, Double);
  cudaCheck(cudaGetLastError());

  forwardSubstitutionKernel<<<1, 1>>>(N, a_float, y_float, rhs, Float);
  cudaCheck(cudaGetLastError());

  forwardSubstitutionKernel<<<1, 1>>>(N, a_half, y_half, rhs, Half);
  cudaCheck(cudaGetLastError());

  // Backward Substitution
  backwardSubstitutionKernel<<<1, 1>>>(N, b_double, y_double, u_double,
                                       super_diag, Double);
  cudaCheck(cudaGetLastError());

  backwardSubstitutionKernel<<<1, 1>>>(N, b_float, y_float, u_float, super_diag,
                                       Float);
  cudaCheck(cudaGetLastError());

  backwardSubstitutionKernel<<<1, 1>>>(N, b_half, y_half, u_half, super_diag,
                                       Half);
  cudaCheck(cudaGetLastError());

  // Integrate
  reimannIntegrationKernel<<<1, 1>>>(N, u_double, p_double, Double);
  cudaCheck(cudaGetLastError());

  reimannIntegrationKernel<<<1, 1>>>(N, u_float, p_float, Float);
  cudaCheck(cudaGetLastError());

  reimannIntegrationKernel<<<1, 1>>>(N, u_half, p_half, Half);
  cudaCheck(cudaGetLastError());

  // Synchronize
  cudaDeviceSynchronize();

  // Compute the backward error
  computeBackwardErrorThomas(N, sub_diag_double, main_diag_double,
                             super_diag_double, rhs_double, a_float, b_float,
                             u_float, ebwd_float);
  computeBackwardErrorThomas(N, sub_diag_double, main_diag_double,
                             super_diag_double, rhs_double, a_half, b_half,
                             u_half, ebwd_half);
  std::cout << *ebwd_float << ", " << static_cast<double>(*ebwd_half)
            << std::endl;

  // Free
  cudaFree(sub_diag);
  cudaFree(main_diag);
  cudaFree(super_diag);
  cudaFree(rhs);

  cudaFree(sub_diag_double);
  cudaFree(main_diag_double);
  cudaFree(super_diag_double);
  cudaFree(rhs_double);

  cudaFree(a_double);
  cudaFree(b_double);
  cudaFree(y_double);
  cudaFree(u_double);
  cudaFree(p_double);

  cudaFree(a_float);
  cudaFree(b_float);
  cudaFree(y_float);
  cudaFree(u_float);
  cudaFree(p_float);

  cudaFree(a_half);
  cudaFree(b_half);
  cudaFree(y_half);
  cudaFree(u_half);
  cudaFree(p_half);
}

void launchStochasticODEExperiment(int N_lower, int bit_shift, int max_shift,
                                   int num_exps, double confidence) {
  int N = N_lower;
  const int width_int = 6;      // For I/O
  const int width_double = 15;  // For I/O

  double *ebwd, *ebwd_model;
  double *ebwd_bound_det, *ebwd_bound_hoeff, *ebwd_bound_bern;
  // Declarations
  half *parameters;  // parameters (always in lower precision to avoid
                     // representation error)

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

  cudaCheck(cudaMallocManaged(&parameters, 2 * num_exps * sizeof(half)));

  for (int ii = 0; ii < max_shift; ii++) {
    std::cout << "Problem size: " << N << std::endl;

    // Compute backward bound (Float)
    ebwd_bound_det[ii] =
        thomasBackwardBound(N - 1, Float, Deterministic, confidence);
    ebwd_bound_hoeff[ii] =
        thomasBackwardBound(N - 1, Float, Hoeffding, confidence);
    ebwd_bound_bern[ii] = thomasBackwardBound(N, Float, Bernstein, confidence);

    // Compute backward bound (Half)
    ebwd_bound_det[max_shift + ii] =
        thomasBackwardBound(N - 1, Half, Deterministic, confidence);
    ebwd_bound_hoeff[max_shift + ii] =
        thomasBackwardBound(N - 1, Half, Hoeffding, confidence);
    ebwd_bound_bern[max_shift + ii] =
        thomasBackwardBound(N - 1, Half, Bernstein, confidence);

    /* std::cout << ebwd_bound_det[ii]  << ", " << ebwd_bound_hoeff[ii] << ", "
     * << ebwd_bound_bern[ii] << ", " */
    /*     << ebwd_bound_det[max_shift + ii]  << ", " <<
     * ebwd_bound_hoeff[max_shift + ii] << ", " << ebwd_bound_bern[max_shift +
     * ii] << std::endl; */

    // Carry experiment of ODE (each experiment is one parameter realization)
    unsigned long long base_seed =
        static_cast<unsigned long long>(std::time(nullptr));
    base_seed += sqrt(ii * 231);
    getODEParameters(num_exps, parameters, base_seed);  // Sample the parameters

    for (int jj = 0; jj < num_exps; jj++) {
      launchODE(N, &parameters[jj], &parameters[num_exps + jj],
                &ebwd[ii * num_exps + jj],
                &ebwd[max_shift * num_exps + ii * num_exps + jj],
                &ebwd_model[ii * num_exps + jj],
                &ebwd_model[max_shift * num_exps + ii * num_exps + jj],
                base_seed);
    }

    N = N << bit_shift;  // Increase problem size
  }

  // Free memory
}
