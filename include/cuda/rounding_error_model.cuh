#ifndef ROUNDING_ERROR_MODEL_CUH
#define ROUNDING_ERROR_MODEL_CUH

#include <curand_kernel.h>

#include "definition.hpp"

/* sample from the rounding error distribution */
__device__ void sample_rounding_error_distribution(const int n,
                                                   double *rounding_error,
                                                   Precision prec,
                                                   BoundModel bound_model,
                                                   curandState *state);

/* compute the unit roundoff */
__device__ __host__ double compute_unit_roundoff(Precision prec);

#endif
