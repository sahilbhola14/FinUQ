#include "bounds.hpp"

#include "gamma.hpp"
#include "probability.hpp"

double dotProductBackwardBound(int N, Precision prec, BoundType btype,
                               double confidence) {
  // Note: confidence is the overall confidence, zeta is used to compute gamma
  double zeta = solveZetaGivenQ(N, confidence);
  return getGamma(N, prec, btype, zeta);
}

double matVecBackwardBound(int N, Precision prec, BoundType btype,
                           double confidence) {
  // Note: For A*x, where A is of size m\times n. Q(\zeta, m\times n) is used.
  // However, for square matrix: Q(\zeta, n^2) is used
  double zeta = solveZetaGivenQ(N * N, confidence);
  return getGamma(N, prec, btype, zeta);
}

double thomasBackwardBound(int N, Precision prec, BoundType btype,
                           double confidence) {
  double zeta = solveZetaGivenQ(7 * N - 6, confidence);
  double gamma_1 = getGamma(1, prec, btype, zeta);
  double gamma_2 = getGamma(1, prec, btype, zeta);
  return 2.0 * gamma_1 + gamma_2 + gamma_1 * gamma_2;
}
