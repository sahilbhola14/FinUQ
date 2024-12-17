#include "ebwd.hpp"

#include <cuda_fp16.h>

#include <cmath>
#include <iostream>

template <typename T>
void computeBackwardErrorDotProduct(double *true_result, T *approx_result,
                                    double *abs_result, double *ebwd) {
  // Equation 7 in the paper
  double delta = *ebwd =
      std::abs(static_cast<double>(*approx_result) - *true_result) /
      *abs_result;
}

template <typename T>
void computeBackwardErrorMatVecMult(int N, double *true_result,
                                    T *approx_result, double *abs_result,
                                    double *ebwd) {
  double error = 0.0;
  for (int i = 0; i < N; i++) {
    error = std::max(error, std::abs(static_cast<double>(approx_result[i]) -
                                     true_result[i]) /
                                abs_result[i]);
  }

  *ebwd = error;
}

template <typename T>
void computeResidual(int N, double *sub_diag, double *main_diag,
                     double *super_diag, double *rhs, T *u, double *residual) {
  const int N_inner = N - 1;
  residual[0] = (main_diag[0] * static_cast<double>(u[0]) +
                 super_diag[0] * static_cast<double>(u[1])) -
                rhs[0];
  for (int ii = 1; ii < N_inner - 1; ii++) {
    residual[ii] = (sub_diag[ii] * static_cast<double>(u[ii - 1]) +
                    main_diag[ii] * static_cast<double>(u[ii]) +
                    super_diag[ii] * static_cast<double>(u[ii + 1])) -
                   rhs[ii];
  }
  residual[N_inner - 1] =
      (sub_diag[N_inner - 1] * static_cast<double>(u[N_inner - 2]) +
       main_diag[N_inner - 1] * static_cast<double>(u[N_inner - 1])) -
      rhs[N_inner - 1];
}

template <typename T>
void computeAbsoluteLUTimesSol(int N, T *a, T *b, double *super_diag, T *u,
                               double *abs_prod) {
  const int N_inner = N - 1;
  double *y = static_cast<double *>(malloc(N_inner * sizeof(double)));
  // Compute U times u
  for (int ii = 0; ii < N_inner - 1; ii++) {
    y[ii] = std::abs(static_cast<double>(b[ii])) *
                std::abs(static_cast<double>(u[ii])) +
            std::abs(super_diag[ii]) * std::abs(static_cast<double>(u[ii + 1]));
  }
  y[N_inner - 1] = std::abs(static_cast<double>(b[N_inner - 1])) *
                   std::abs(static_cast<double>(u[N_inner - 1]));
  // Compute L times y
  abs_prod[0] = y[0];
  for (int ii = 1; ii < N_inner; ii++) {
    abs_prod[ii] = std::abs(static_cast<double>(a[ii])) * std::abs(y[ii - 1]) +
                   std::abs(y[ii]);
  }

  free(y);
}

template <typename T>
void computeBackwardErrorThomas(int N, double *sub_diag, double *main_diag,
                                double *super_diag, double *rhs, T *a, T *b,
                                T *u, double *ebwd) {
  const int N_inner = N - 1;
  // Compute residual
  double *residual = static_cast<double *>(malloc(N_inner * sizeof(double)));
  double *abs_prod = static_cast<double *>(malloc(N_inner * sizeof(double)));
  double error = 0.0;
  computeResidual(N, sub_diag, main_diag, super_diag, rhs, u, residual);
  // Compute the absolute LU*u
  computeAbsoluteLUTimesSol(N, a, b, super_diag, u, abs_prod);

  for (int ii = 0; ii < N_inner; ii++) {
    error = std::max(error, std::abs(residual[ii]) / abs_prod[ii]);
  }

  *ebwd = error;
}

// Template compilation
template void computeBackwardErrorDotProduct(double *, float *, double *,
                                             double *);
template void computeBackwardErrorDotProduct(double *, half *, double *,
                                             double *);

template void computeBackwardErrorMatVecMult(int, double *, float *, double *,
                                             double *);
template void computeBackwardErrorMatVecMult(int, double *, half *, double *,
                                             double *);

template void computeBackwardErrorThomas(int, double *, double *, double *,
                                         double *, float *, float *, float *,
                                         double *);
template void computeBackwardErrorThomas(int, double *, double *, double *,
                                         double *, half *, half *, half *,
                                         double *);

template void computeAbsoluteLUTimesSol(int, float *, float *, double *,
                                        float *, double *);
template void computeAbsoluteLUTimesSol(int, half *, half *, double *, half *,
                                        double *);
