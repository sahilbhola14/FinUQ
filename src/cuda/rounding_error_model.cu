#include <iostream>

#include "definition.hpp"
#include "rounding_error_model.cuh"

/* compute the unit roundoff */
__device__ __host__ double compute_unit_roundoff(Precision prec) {
  double base, precision;
  double urd = 0.0;
  if (prec == Half) {
    base = 2.0;
    precision = 11.0;
  } else if (prec == Single) {
    base = 2.0;
    precision = 24.0;
  } else if (prec == Double) {
    base = 2.0;
    precision = 53.0;
  } else {
    printf("<Cuda Error> Invalid precision");
    return 0.0;
  }
  urd = pow(base, -(precision - 1.0)) / 2.0;
  return urd;
}

__device__ void sample_rounding_error_beta_model(const int n,
                                                 double *rounding_error,
                                                 Precision prec,
                                                 curandState *state) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
}

/* sample delta~U(-urd, urd) */
__device__ void sample_rounding_error_uniform_model(const int n,
                                                    double *rounding_error,
                                                    Precision prec,
                                                    curandState *state) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* sample rounding error */
  for (int i = 0; i < n; i++) {
    /* sample from uniform distribution */
    double r = curand_uniform_double(state);
    /* sample from the uniform distribution */
    rounding_error[i] = -urd + 2.0 * urd * r;
  }
}

/* sample rounding error distribution for a given precision */
__device__ void sample_rounding_error_distribution(const int n,
                                                   double *rounding_error,
                                                   Precision prec,
                                                   BoundModel bound_model,
                                                   curandState *state) {
  switch (bound_model) {
    case Uniform:
      sample_rounding_error_uniform_model(n, rounding_error, prec, state);
      break;
    case Beta:
      sample_rounding_error_beta_model(n, rounding_error, prec, state);
      break;
    default:
      printf("<Cuda error>: Invalid bound model");
      return;
  }
}
