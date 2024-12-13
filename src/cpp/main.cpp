#include <iostream>

#include "dotProduct.cuh"
#include "gamma.hpp"
#include "matVecMult.cuh"

int main() {
  // Compare Gamma [Figure 1]
  /* compareGamma(2, Float, "gamma_zeta_0.9_fp32.txt", 2, 13, 0.9); */
  /* compareGamma(2, Float, "gamma_zeta_0.99_fp32.txt", 2, 13, 0.99); */
  /* compareGamma(2, Half, "gamma_zeta_0.9_fp16.txt", 2, 13, 0.9); */
  /* compareGamma(2, Half, "gamma_zeta_0.99_fp16.txt", 2, 13, 0.99); */

  // Dot Product [Figure 2]
  /* launchDotProductExperiment(2<<5, 1, 15); */
  // Matrix-Vector Product
  launchMatVecMultExperiment(2 << 5, 1, 9);
  return 0;
}
