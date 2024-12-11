#include "gamma.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "utils.hpp"

double getGamma(int N, Precision prec, BoundType btype, double confidence) {
  double urd = computeUnitRoundOff(prec);
  double gamma;

  if (btype == Deterministic) {
    if (N * urd < 1.0) {
      gamma = (N * urd) / (1.0 - N * urd);
    } else {
      gamma = std::numeric_limits<double>::infinity();
    }
  } else if (btype == Hoeffding) {
    double lambda =
        (1.0 / (1.0 - urd)) * sqrt(-2 * log((1 - confidence) / 2.0));
    double t = lambda * sqrt(N) * urd;
    gamma = t + (N * urd * urd) / (1.0 - urd);
  } else if (btype == Bernstein) {
    double mu = (-2.0 * urd + (-1.0 + urd) * log(1.0 - urd) +
                 (1.0 + urd) * log(1.0 + urd)) /
                (2.0 * urd);
    double kappa = -1.0 + pow(urd, 2);
    double c = log(1.0 + urd);
    double var =
        (4 * pow(urd, 2) + kappa * (pow(log(1.0 - urd), 2) -
                                    2.0 * log(1.0 - urd) * log(1.0 + urd) +
                                    pow(log(1.0 + urd), 2))) /
        (4 * pow(urd, 2));
    double log_term = log((1 - confidence) / 2.0);
    double t = (1.0 / 3.0) * (-c * log_term + sqrt(pow(c * log_term, 2) -
                                                   18.0 * N * log_term * var));
    gamma = t + N * abs(mu);
  } else {
    std::invalid_argument("Invalid bound type");
  }
  return gamma;
}

void compareGamma(int N_lower, Precision prec, int bit_shift, int max_shift,
                  double confidence) {
  int N = N_lower;
  double gamma_deterministic, gamma_hoeffding, gamma_bernstein;
  const int width_int = 6;
  const int width_double = 15;
  std::string filename =
      "gamma_confidence_" + std::to_string(confidence) + ".txt";
  std::ofstream outfile(filename);
  if (!outfile) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  outfile << "Description: Comparison of gamma for a given confidence ("
          << confidence << ")" << std::endl;
  outfile
      << "Variables: N, gamma_deterministic, gamma_hoeffding, gamma_berinstein"
      << std::endl;

  for (int i = 0; i < max_shift; i++) {
    /* printf("--------\n"); */
    /* printf("N: %d\n", N); */
    gamma_deterministic = getGamma(N, prec, Deterministic, confidence);
    gamma_hoeffding = getGamma(N, prec, Hoeffding, confidence);
    gamma_bernstein = getGamma(N, prec, Bernstein, confidence);
    outfile << std::setw(width_int) << N << "," << std::setw(width_double)
            << std::scientific << std::setprecision(8) << gamma_deterministic
            << "," << std::setw(width_double) << gamma_hoeffding << ","
            << std::setw(width_double) << gamma_bernstein << std::endl;
    N = N << bit_shift;
  }
  std::cout << "Data written to : " << filename << std::endl;
  outfile.close();
}
