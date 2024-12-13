#include "probability.hpp"

double getQProb(double zeta, int N) { return 1.0 - N * (1 - zeta); }

double solveZetaGivenQ(int N, double targetQ) {
  // Solves for zeta given Q
  return 1.0 - ((1.0 - targetQ) / N);
}

double dotProductProb(double zeta, int N) { return getQProb(zeta, N); }
