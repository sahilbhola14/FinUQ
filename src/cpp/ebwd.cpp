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

// Template compilation
template void computeBackwardErrorDotProduct(double *, float *, double *,
                                             double *);
template void computeBackwardErrorDotProduct(double *, half *, double *,
                                             double *);

template void computeBackwardErrorMatVecMult(int, double *, float *, double *,
                                             double *);
template void computeBackwardErrorMatVecMult(int, double *, half *, double *,
                                             double *);
