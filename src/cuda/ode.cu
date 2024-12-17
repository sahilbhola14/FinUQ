#include <assert.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "bounds.hpp"
#include "checks.cuh"
#include "config.hpp"
#include "convert.cuh"
#include "definition.hpp"
#include "ebwd.hpp"
#include "efwd.hpp"
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

__global__ void getModelDecompositionKernel(const int N, half *sub_diag,
                                            half *main_diag, half *super_diag,
                                            double *a, double *b, double urd) {
  const int N_inner = N - 1;
  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  double eta, psi, xi;
  unsigned long long seed = 5678ULL;
  curandState state;
  curand_init(seed, gid, 0, &state);

  b[0] = static_cast<double>(main_diag[0]);

  for (int ii = 1; ii < N_inner; ii++) {
    eta = 1.0 + sampleRelativeErrorKernel(urd, &state);
    psi = 1.0 + sampleRelativeErrorKernel(urd, &state);
    xi = 1.0 + sampleRelativeErrorKernel(urd, &state);

    a[ii] =
        __ddiv_rn(static_cast<double>(sub_diag[ii]), __dmul_rn(b[ii - 1], eta));
    b[ii] = __ddiv_rn(
        __dsub_rn(
            static_cast<double>(main_diag[ii]),
            __dmul_rn(a[ii],
                      __dmul_rn(static_cast<double>(super_diag[ii - 1]), psi))),
        xi);
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

__global__ void forwardModelSubstitutionKernel(const int N, double *a,
                                               double *y, half *rhs,
                                               double urd) {
  const int N_inner = N - 1;

  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  double eta, xi;
  unsigned long long seed = 1435ULL;
  curandState state;
  curand_init(seed, gid, 0, &state);

  y[0] = static_cast<double>(rhs[0]);

  for (int ii = 1; ii < N_inner; ii++) {
    eta = 1.0 + sampleRelativeErrorKernel(urd, &state);
    xi = 1.0 + sampleRelativeErrorKernel(urd, &state);

    y[ii] = __ddiv_rn(__dsub_rn(static_cast<double>(rhs[ii]),
                                __dmul_rn(a[ii], __dmul_rn(y[ii - 1], eta))),
                      xi);
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

__global__ void backwardModelSubstitutionKernel(const int N, double *b,
                                                double *y, double *u,
                                                half *super_diag, double urd) {
  const int N_inner = N - 1;

  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  double eta, psi, xi, chi, xi_chi;
  unsigned long long seed = 3456ULL;
  curandState state;
  curand_init(seed, gid, 0, &state);

  eta = 1.0 + sampleRelativeErrorKernel(urd, &state);

  u[N_inner - 1] = __ddiv_rn(y[N_inner - 1], __dmul_rn(b[N_inner - 1], eta));

  for (int ii = N_inner - 2; ii >= 0; ii--) {
    psi = 1.0 + sampleRelativeErrorKernel(urd, &state);
    xi = 1.0 + sampleRelativeErrorKernel(urd, &state);
    chi = 1.0 + sampleRelativeErrorKernel(urd, &state);
    xi_chi = __dmul_rn(xi, chi);

    u[ii] = __ddiv_rn(
        __dsub_rn(y[ii], __dmul_rn(static_cast<double>(super_diag[ii]),
                                   __dmul_rn(u[ii + 1], psi))),
        __dmul_rn(b[ii], xi_chi));
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

__global__ void reimannModelIntegrationKernel(const int N, double *u, double *p,
                                              double urd) {
  const int N_inner = N - 1;
  const double dx = static_cast<double>(
      static_cast<half>(1.0 / N));  // Assumes no representation error
  double summation = static_cast<double>(0.0);

  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;
  double eta, xi;
  unsigned long long seed = 3456ULL;
  curandState state;
  curand_init(seed, gid, 0, &state);

  for (int ii = 0; ii < N_inner; ii++) {
    eta = 1.0 + sampleRelativeErrorKernel(urd, &state);
    summation = __dmul_rn(__dadd_rn(summation, u[ii]), eta);
  }

  xi = 1.0 + sampleRelativeErrorKernel(urd, &state);
  summation = __dmul_rn(summation, __dmul_rn(dx, xi));

  *p = summation;
}

void computeAnalyticalSolution(half *theta_1, half *theta_2, double *p) {
  double numerator, denominator;
  double t1 = static_cast<double>(*theta_1);
  double t2 = static_cast<double>(*theta_2);
  numerator = 25.0 * t2 * t2 * (-2.0 * t1 + (2.0 + t1) * log(1 + t1));
  denominator = t1 * t1 * log(1 + t1);
  *p = numerator / denominator;
}

void launchODE(const int N, half *theta_1, half *theta_2,
               double *ebwd_thomas_float, double *ebwd_thomas_half,
               double *ebwd_thomas_model_float, double *ebwd_thomas_model_half,
               double *efwd_thomas_model_float, double *efwd_thomas_model_half,
               double *efwd_thomas_float, double *efwd_thomas_half,
               double *efwd_qoi_float, double *efwd_qoi_half,
               double *efwd_qoi_model_float, double *efwd_qoi_model_half,
               double *efwd_qoi_bound_det_float,
               double *efwd_qoi_bound_det_half,
               double *efwd_qoi_bound_hoeff_float,
               double *efwd_qoi_bound_hoeff_half,
               double *efwd_qoi_bound_bern_float,
               double *efwd_qoi_bound_bern_half, double *qoi_double,
               double *qoi_float, double *qoi_half, double confidence) {
  double dx = 1.0 / N;  // gridSize
  double *C_ls_float, *C_ls_half, *C_ls_float_model, *C_ls_half_model;

  half *sub_diag, *main_diag, *super_diag,
      *rhs;  // Always in lower precision to avoid represenation error

  double *sub_diag_double, *main_diag_double, *super_diag_double, *rhs_double;

  double *a_double, *b_double, *y_double, *u_double, *p_double;
  float *a_float, *b_float, *y_float, *u_float, *p_float;
  half *a_half, *b_half, *y_half, *u_half, *p_half;

  double *a_float_model, *b_float_model, *y_float_model, *u_float_model,
      *p_float_model;
  double *a_half_model, *b_half_model, *y_half_model, *u_half_model,
      *p_half_model;

  double urd_float = computeUnitRoundOff(Float);
  double urd_half = computeUnitRoundOff(Half);

  const int N_inner = N - 1;

  dim3 blockDim = config::blockSize;
  dim3 gridDim =
      getGridSize(blockDim.x, N_inner);  // Map inner elements to each thread

  // Allocation

  C_ls_float = static_cast<double *>(malloc(sizeof(double)));
  C_ls_half = static_cast<double *>(malloc(sizeof(double)));

  C_ls_float_model = static_cast<double *>(malloc(sizeof(double)));
  C_ls_half_model = static_cast<double *>(malloc(sizeof(double)));

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

  cudaCheck(cudaMallocManaged(&a_half_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&b_half_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_half_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&u_half_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&p_half_model, sizeof(double)));

  cudaCheck(cudaMallocManaged(&a_float_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&b_float_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&y_float_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&u_float_model, N_inner * sizeof(double)));
  cudaCheck(cudaMallocManaged(&p_float_model, sizeof(double)));

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

  getModelDecompositionKernel<<<1, 1>>>(N, sub_diag, main_diag, super_diag,
                                        a_float_model, b_float_model,
                                        urd_float);
  cudaCheck(cudaGetLastError());

  getModelDecompositionKernel<<<1, 1>>>(N, sub_diag, main_diag, super_diag,
                                        a_half_model, b_half_model, urd_half);
  cudaCheck(cudaGetLastError());

  // Forward Substitution
  forwardSubstitutionKernel<<<1, 1>>>(N, a_double, y_double, rhs, Double);
  cudaCheck(cudaGetLastError());

  forwardSubstitutionKernel<<<1, 1>>>(N, a_float, y_float, rhs, Float);
  cudaCheck(cudaGetLastError());

  forwardSubstitutionKernel<<<1, 1>>>(N, a_half, y_half, rhs, Half);
  cudaCheck(cudaGetLastError());

  forwardModelSubstitutionKernel<<<1, 1>>>(N, a_float_model, y_float_model, rhs,
                                           urd_float);
  cudaCheck(cudaGetLastError());

  forwardModelSubstitutionKernel<<<1, 1>>>(N, a_half_model, y_half_model, rhs,
                                           urd_half);
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

  backwardModelSubstitutionKernel<<<1, 1>>>(
      N, b_float_model, y_float_model, u_float_model, super_diag, urd_float);
  cudaCheck(cudaGetLastError());

  backwardModelSubstitutionKernel<<<1, 1>>>(N, b_half_model, y_half_model,
                                            u_half_model, super_diag, urd_half);
  cudaCheck(cudaGetLastError());

  // Integrate
  reimannIntegrationKernel<<<1, 1>>>(N, u_double, p_double, Double);
  cudaCheck(cudaGetLastError());

  reimannIntegrationKernel<<<1, 1>>>(N, u_float, p_float, Float);
  cudaCheck(cudaGetLastError());

  reimannIntegrationKernel<<<1, 1>>>(N, u_half, p_half, Half);
  cudaCheck(cudaGetLastError());

  reimannModelIntegrationKernel<<<1, 1>>>(N, u_float_model, p_float_model,
                                          urd_float);
  cudaCheck(cudaGetLastError());

  reimannModelIntegrationKernel<<<1, 1>>>(N, u_half_model, p_half_model,
                                          urd_half);
  cudaCheck(cudaGetLastError());

  // Synchronize
  cudaDeviceSynchronize();

  // Store qoi
  *qoi_double = *p_double;
  *qoi_float = static_cast<double>(*p_float);
  *qoi_half = static_cast<double>(*p_half);

  // Compute the backward error for Thomas
  computeBackwardErrorThomas(N, sub_diag_double, main_diag_double,
                             super_diag_double, rhs_double, a_float, b_float,
                             u_float, ebwd_thomas_float);

  computeBackwardErrorThomas(N, sub_diag_double, main_diag_double,
                             super_diag_double, rhs_double, a_half, b_half,
                             u_half, ebwd_thomas_half);

  computeBackwardErrorThomas(
      N, sub_diag_double, main_diag_double, super_diag_double, rhs_double,
      a_float_model, b_float_model, u_float_model, ebwd_thomas_model_float);

  computeBackwardErrorThomas(
      N, sub_diag_double, main_diag_double, super_diag_double, rhs_double,
      a_half_model, b_half_model, u_half_model, ebwd_thomas_model_half);

  /* std::cout << *ebwd_thomas_float << ", " << *ebwd_thomas_half << ", " <<
   * *ebwd_thomas_model_float << ", " << *ebwd_thomas_model_half << ", " <<
   * std::endl; */

  // Comptue the forward error for Thomas
  computeForwardErrorThomas(N, sub_diag_double, main_diag_double,
                            super_diag_double, rhs_double, a_float, b_float,
                            u_float, ebwd_thomas_float, efwd_thomas_float,
                            C_ls_float);

  computeForwardErrorThomas(
      N, sub_diag_double, main_diag_double, super_diag_double, rhs_double,
      a_half, b_half, u_half, ebwd_thomas_half, efwd_thomas_half, C_ls_half);

  computeForwardErrorThomas(
      N, sub_diag_double, main_diag_double, super_diag_double, rhs_double,
      a_float_model, b_float_model, u_float_model, ebwd_thomas_model_float,
      efwd_thomas_model_float, C_ls_float_model);

  computeForwardErrorThomas(N, sub_diag_double, main_diag_double,
                            super_diag_double, rhs_double, a_half_model,
                            b_half_model, u_half_model, ebwd_thomas_model_half,
                            efwd_thomas_model_half, C_ls_half_model);

  /* std::cout << *efwd_thomas_float << ", " << *efwd_thomas_half << ", " <<
   * *efwd_thomas_model_float << ", " << *efwd_thomas_model_half << ", " <<
   * std::endl; */

  // Compute the forward error for the Qoi
  *efwd_qoi_float = std::abs(static_cast<double>(*p_float) - *p_double);
  *efwd_qoi_half = std::abs(static_cast<double>(*p_half) - *p_double);
  *efwd_qoi_model_float =
      std::abs(static_cast<double>(*p_float_model) - *p_double);
  *efwd_qoi_model_half =
      std::abs(static_cast<double>(*p_half_model) - *p_double);

  // Compute the forward error bound for Qoi
  computeForwardErrorQoi(N, u_float, C_ls_float, efwd_qoi_bound_det_float,
                         Deterministic, Float, confidence);
  computeForwardErrorQoi(N, u_float, C_ls_float, efwd_qoi_bound_hoeff_float,
                         Hoeffding, Float, confidence);
  computeForwardErrorQoi(N, u_float, C_ls_float, efwd_qoi_bound_bern_float,
                         Bernstein, Float, confidence);

  computeForwardErrorQoi(N, u_half, C_ls_half, efwd_qoi_bound_det_half,
                         Deterministic, Half, confidence);
  computeForwardErrorQoi(N, u_half, C_ls_half, efwd_qoi_bound_hoeff_half,
                         Hoeffding, Half, confidence);
  computeForwardErrorQoi(N, u_half, C_ls_half, efwd_qoi_bound_bern_half,
                         Bernstein, Half, confidence);

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

void saveMat(const int iMax, const int jMax, double *data,
             std::ofstream &outfile) {
  const int width_int = 6;      // For I/O
  const int width_double = 15;  // For I/O

  outfile << std::setw(width_int + 1) << "    ";
  for (int ii = 0; ii < jMax; ii++) {
    for (int jj = 0; jj < iMax; jj++) {
      outfile << std::setw(width_double) << std::scientific
              << std::setprecision(8) << data[jj * jMax + ii];
      if (jj < iMax - 1) outfile << ", ";
    }
    outfile << std::endl;
    if (ii < jMax - 1) outfile << std::setw(width_int + 1) << "    ";
  }
}

void launchStochasticODEExperiment(int N_lower, int bit_shift, int max_shift,
                                   int num_exps, double confidence) {
  int N = N_lower;
  const int width_int = 6;      // For I/O
  const int width_double = 15;  // For I/O
  double *p, *p_analytical;

  double *ebwd_thomas, *ebwd_thomas_model;
  double *efwd_thomas, *efwd_qoi;
  double *efwd_thomas_model, *efwd_qoi_model;

  double *efwd_qoi_bound_det, *efwd_qoi_bound_hoeff, *efwd_qoi_bound_bern;

  double *ebwd_bound_det, *ebwd_bound_hoeff, *ebwd_bound_bern;
  // Declarations
  half *parameters;  // parameters (always in lower precision to avoid
                     // representation error)

  p = static_cast<double *>(malloc(3 * max_shift * num_exps * sizeof(double)));
  p_analytical =
      static_cast<double *>(malloc(max_shift * num_exps * sizeof(double)));

  ebwd_bound_det =
      static_cast<double *>(malloc(2 * max_shift * sizeof(double)));
  ebwd_bound_hoeff =
      static_cast<double *>(malloc(2 * max_shift * sizeof(double)));
  ebwd_bound_bern =
      static_cast<double *>(malloc(2 * max_shift * sizeof(double)));

  ebwd_thomas =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));
  ebwd_thomas_model =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));

  efwd_thomas_model =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));
  efwd_qoi_model =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));

  efwd_thomas =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));

  efwd_qoi =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));
  efwd_qoi_bound_det =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));
  efwd_qoi_bound_hoeff =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));
  efwd_qoi_bound_bern =
      static_cast<double *>(malloc(2 * max_shift * num_exps * sizeof(double)));

  cudaCheck(cudaMallocManaged(&parameters, 2 * num_exps * sizeof(half)));

  // each experiment is one parameter realization (Same parameters for all
  // discretizations)
  unsigned long long base_seed =
      static_cast<unsigned long long>(std::time(nullptr));
  base_seed += sqrt(231);
  getODEParameters(num_exps, parameters, base_seed);  // Sample the parameters

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

    for (int jj = 0; jj < num_exps; jj++) {
      // Compute the analytical solution
      computeAnalyticalSolution(&parameters[jj], &parameters[num_exps + jj],
                                &p_analytical[ii * num_exps + jj]);

      // Compute ODE solution
      launchODE(
          N, &parameters[jj], &parameters[num_exps + jj],
          &ebwd_thomas[ii * num_exps + jj],  // Backward error (fp32)
          &ebwd_thomas[max_shift * num_exps + ii * num_exps +
                       jj],  // Backward error (fp16)
          &ebwd_thomas_model[ii * num_exps +
                             jj],  // Backward model error (fp32)
          &ebwd_thomas_model[max_shift * num_exps + ii * num_exps +
                             jj],  // Backward model error (fp16)
          &efwd_thomas_model[ii * num_exps + jj],  // Forward model error (fp32)
          &efwd_thomas_model[max_shift * num_exps + ii * num_exps +
                             jj],            // Forward model error (fp16)
          &efwd_thomas[ii * num_exps + jj],  // Forward error (fp32)
          &efwd_thomas[max_shift * num_exps + ii * num_exps +
                       jj],               // Forward error (fp16)
          &efwd_qoi[ii * num_exps + jj],  // Forward error qoi (fp32)
          &efwd_qoi[max_shift * num_exps + ii * num_exps +
                    jj],  // Forward error qoi (fp16)
          &efwd_qoi_model[ii * num_exps +
                          jj],  // Forward model error qoi (fp32)
          &efwd_qoi_model[max_shift * num_exps + ii * num_exps +
                          jj],  // Forward model error qoi (fp16)
          &efwd_qoi_bound_det[ii * num_exps +
                              jj],  // Forward bound deterministic (fp32)
          &efwd_qoi_bound_det[max_shift * num_exps + ii * num_exps +
                              jj],  // Forward bound deterministic (fp16)
          &efwd_qoi_bound_hoeff[ii * num_exps +
                                jj],  // Forward bound hoeffding (fp32)
          &efwd_qoi_bound_hoeff[max_shift * num_exps + ii * num_exps +
                                jj],  // Forward bound hoeffding (fp16)
          &efwd_qoi_bound_bern[ii * num_exps +
                               jj],  // Forward bound bernstein (fp32)
          &efwd_qoi_bound_bern[max_shift * num_exps + ii * num_exps +
                               jj],  // Forward bound bernstein (fp16)
          &p[ii * num_exps + jj],    // Double qoi result
          &p[max_shift * num_exps + ii * num_exps + jj],  // Float qoi result
          &p[2 * max_shift * num_exps + ii * num_exps + jj],  // Half qoi result
          confidence);
    }

    N = N << bit_shift;  // Increase problem size
  }

  /* // Save the data */
  std::ofstream outfile;
  std::string filename;

  // Backward Error Thomas (FP32) with and Without Model
  filename = "thomas_ebwd_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward Error Thomas (FP32)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, ebwd_thomas, outfile);
  outfile.close();

  filename = "thomas_ebwd_model_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward Model Error Thomas (FP32)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, ebwd_thomas_model, outfile);
  outfile.close();

  // Backward Error Thomas (FP16) with and Without Model
  filename = "thomas_ebwd_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward Error Thomas (FP16)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, ebwd_thomas + max_shift * num_exps, outfile);
  outfile.close();

  filename = "thomas_ebwd_model_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward Model Error Thomas (FP16)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, ebwd_thomas_model + max_shift * num_exps,
          outfile);
  outfile.close();

  // Forward Error Thomas (FP32) with and Without Model
  filename = "thomas_efwd_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Thomas (FP32)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_thomas, outfile);
  outfile.close();

  filename = "thomas_efwd_model_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Model Error Thomas (FP32)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_thomas_model, outfile);
  outfile.close();

  // Forward Error Thomas (FP16) with and Without Model
  filename = "thomas_efwd_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Thomas (FP16)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_thomas + max_shift * num_exps, outfile);
  outfile.close();

  filename = "thomas_efwd_model_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Model Error Thomas (FP16)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_thomas_model + max_shift * num_exps,
          outfile);
  outfile.close();

  // Forward Error QoI (FP32) with and Without Model
  filename = "qoi_efwd_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error QoI (FP32)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi, outfile);
  outfile.close();

  filename = "qoi_efwd_model_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Model Error QoI (FP32)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_model, outfile);
  outfile.close();

  // Forward Error QoI (FP16) with and Without Model
  filename = "qoi_efwd_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error QoI (FP16)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi + max_shift * num_exps, outfile);
  outfile.close();

  filename = "qoi_efwd_model_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Model Error QoI (FP16)" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_model + max_shift * num_exps, outfile);
  outfile.close();

  // Backward Error Bound Thomas (FP32 and FP16)
  filename = "thomas_ebwd_bound_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward Error Bound Thomas (FP32)" << std::endl;
  outfile << "Rows: Deterministic, Hoeffding, Bersnstein" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, 1, ebwd_bound_det, outfile);
  saveMat(max_shift, 1, ebwd_bound_hoeff, outfile);
  saveMat(max_shift, 1, ebwd_bound_bern, outfile);
  outfile.close();

  filename = "thomas_ebwd_bound_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Backward Error Bound Thomas (FP16)" << std::endl;
  outfile << "Rows: Deterministic, Hoeffding, Bersnstein" << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, 1, ebwd_bound_det + max_shift, outfile);
  saveMat(max_shift, 1, ebwd_bound_hoeff + max_shift, outfile);
  saveMat(max_shift, 1, ebwd_bound_bern + max_shift, outfile);
  outfile.close();

  // Forward Error Bound QoI (FP32)
  filename = "qoi_efwd_bound_deterministic_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Deterministic Bound QoI (FP32)"
          << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_bound_det, outfile);
  outfile.close();

  filename = "qoi_efwd_bound_hoeffding_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Hoeffding Bound QoI (FP32)"
          << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_bound_hoeff, outfile);
  outfile.close();

  filename = "qoi_efwd_bound_bernstein_fp32.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Bernstein Bound QoI (FP32)"
          << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_bound_bern, outfile);
  outfile.close();

  // Forward Error Bound QoI (FP16)
  filename = "qoi_efwd_bound_deterministic_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Deterministic Bound QoI (FP16)"
          << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_bound_det + max_shift * num_exps,
          outfile);
  outfile.close();

  filename = "qoi_efwd_bound_hoeffding_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Hoeffding Bound QoI (FP16)"
          << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_bound_hoeff + max_shift * num_exps,
          outfile);
  outfile.close();

  filename = "qoi_efwd_bound_bernstein_fp16.txt";
  outfile.open(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Forward Error Bernstein Bound QoI (FP16)"
          << std::endl;
  outfile << std::setw(width_int) << "N: ";
  N = N_lower;
  for (int ii = 0; ii < max_shift; ii++) {
    outfile << std::setw(width_double) << N;
    if (ii < max_shift - 1) outfile << ", ";
    N = N << bit_shift;
  }
  outfile << std::endl;
  saveMat(max_shift, num_exps, efwd_qoi_bound_bern + max_shift * num_exps,
          outfile);
  outfile.close();

  // Free memory
  free(ebwd_thomas);
  free(ebwd_thomas_model);
  free(efwd_thomas);
  free(efwd_thomas_model);

  free(efwd_qoi);
  free(efwd_qoi_model);

  free(ebwd_bound_det);
  free(ebwd_bound_hoeff);
  free(ebwd_bound_bern);

  free(p);
}
