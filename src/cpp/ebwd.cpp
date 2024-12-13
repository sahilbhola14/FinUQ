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

// Template compilation
template void computeBackwardErrorDotProduct(double *, float *, double *,
                                             double *);
template void computeBackwardErrorDotProduct(double *, half *, double *,
                                             double *);
