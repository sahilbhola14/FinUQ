#include <iostream>

#include "dotProduct.cuh"
#include "gamma.hpp"

int main() {
  // Compare Gamma
  compareGamma(2, Float);
  /* compareGamma(2, Half); */
  // Dot product experiments
  /* launchDotProductExperiment(); */
  return 0;
}
