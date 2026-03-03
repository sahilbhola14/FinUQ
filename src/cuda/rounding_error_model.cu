#include <math.h>

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

// Marsaglia-Tsang method for Gamma distribution
__device__ double curand_gamma_double(curandState *state, double shape,
                                      double scale) {
  if (shape < 1.0) {
    // For shape < 1, use Weibull transformation
    return scale * pow(-log(curand_uniform_double(state)), 1.0 / shape);
  }

  double d = shape - 1.0 / 3.0;
  double c = 1.0 / sqrt(9.0 * d);

  while (true) {
    double z = curand_normal_double(state);
    double v = 1.0 + c * z;

    if (v <= 0.0) continue;

    v = v * v * v;
    double u = curand_uniform_double(state);

    // Quick accept
    if (u < 1.0 - 0.0331 * z * z * z * z) {
      return scale * d * v;
    }

    // Slow accept/reject
    if (log(u) < 2.0 * log(0.5 * z) + log(d) + log(v) - 1.2) {
      return scale * d * v;
    }
  }
}

__device__ void sample_rounding_error_beta_model(const int n,
                                                 double *rounding_error,
                                                 Precision prec,
                                                 const double beta_dist_alpha,
                                                 const double beta_dist_beta,
                                                 curandState *state) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  double L = log1p(urd) - log1p(-urd);
  /* sample rounding error */
  for (int i = 0; i < n; i++) {
    /* sample from beta distribution */
    double x = curand_gamma_double(state, beta_dist_alpha, 1.0);
    double y = curand_gamma_double(state, beta_dist_beta, 1.0);
    double z = x / (x + y);  // sample from Beta(alpha, beta)
    /* sample delta = exp(Y) - 1, Y~log(1-urd) + log(1+urd/1-urd) * B(alpha,
     * beta) */
    rounding_error[i] = exp(log1p(-urd) + L * z) - 1.0;
  }
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
    double z = curand_uniform_double(state);
    /* sample from the uniform distribution U(-urd, urd)*/
    rounding_error[i] = -urd + 2.0 * urd * z;
  }
}

/* sample rounding error distribution for a given precision */
__device__ void sample_rounding_error_distribution(
    const int n, double *rounding_error, Precision prec, BoundModel bound_model,
    const double beta_dist_alpha, const double beta_dist_beta,
    curandState *state) {
  switch (bound_model) {
    case Uniform:
      sample_rounding_error_uniform_model(n, rounding_error, prec, state);
      break;
    case Beta:
      sample_rounding_error_beta_model(n, rounding_error, prec, beta_dist_alpha,
                                       beta_dist_beta, state);
      break;
    default:
      printf("<Cuda error>: Invalid bound model");
      return;
  }
}
