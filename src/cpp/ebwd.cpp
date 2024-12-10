#include "ebwd.hpp"

#include <iostream>

template <typename T>
void computeBacwardErrorDotProduct(double *true_result, T *approx_result,
                                   double *abs_result, double *ebwd) {
  // Equation 7 in the paper
  *ebwd = abs(static_cast<double>(*approx_result) - *true_result) / *abs_result;
}
