#include <cuda_fp16.h>

#include <ctime>
#include <iostream>

#include "bounds.hpp"
#include "definition.hpp"
#include "matVecMult.cuh"

void launchRecursiveMatVecMult(const int N, double *ebwd_float,
                               double *ebwd_half, double *ebwd_float_model,
                               double *ebwd_half_model,
                               unsigned long long seed) {}

void launchMatVectMultExperiment(int N_lower, int bit_shift, int max_shift,
                                 int num_exps, double confidence) {
  // Square matrix-vector product
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
        matVecBackwardBound(N, Float, Deterministic, confidence);
    ebwd_bound_hoeff[ii] = matVecBackwardBound(N, Float, Hoeffding, confidence);
    ebwd_bound_bern[ii] = matVecBackwardBound(N, Float, Bernstein, confidence);

    // Compute backward bound (Half)
    ebwd_bound_det[max_shift + ii] =
        matVecBackwardBound(N, Half, Deterministic, confidence);
    ebwd_bound_hoeff[max_shift + ii] =
        matVecBackwardBound(N, Half, Hoeffding, confidence);
    ebwd_bound_bern[max_shift + ii] =
        matVecBackwardBound(N, Half, Bernstein, confidence);

    printf("Det: %.5e Higham: %.5e Bern: %.5e\n", ebwd_bound_det[ii],
           ebwd_bound_hoeff[ii], ebwd_bound_bern[ii]);

    // Carry experiment of matrix-vector products
    for (int jj = 0; jj < num_exps; jj++) {
      // Experiment seed
      unsigned long long base_seed =
          static_cast<unsigned long long>(std::time(nullptr));
      base_seed += sqrt(jj * 354);  // Experimental seed
      base_seed += sqrt(ii * 231);  // Problem size seed

      // Single sequential experiment
    }

    // Increase the Matrix/vector size
    N = N << bit_shift;
  }
}
