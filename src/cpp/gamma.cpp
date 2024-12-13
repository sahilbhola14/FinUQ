#include "gamma.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "utils.hpp"

double getLogDistMean(double urd, double kappa) {
  double numerator, denominator;
  numerator = (2 * urd * (-2 + kappa) -
               (-1 + urd) * (-2 + kappa + urd * kappa) * log(1 - urd) +
               (1 + urd) * (2 + (-1 + urd) * kappa) * log(1 + urd));
  denominator = 4 * urd;
  return numerator / denominator;
}

double getLogDistVar(double urd, double kappa) {
  /* Define logarithm functions */
  double log1_minus_a = log(1 - urd);
  double log1_plus_a = log(1 + urd);

  /* Compute individual terms */
  double term1 = 2 * urd * (-2 + kappa);
  double term2 = (-1 + urd) * (-2 + kappa + urd * kappa) * log1_minus_a;
  double term3 = (1 + urd) * (2 + (-1 + urd) * kappa) * log1_plus_a;

  double numerator1 = term1 - term2 + term3;
  double numerator1_squared = pow(numerator1, 2);

  double term4 = urd * (8 - 6 * kappa);
  double term5 = (-1 + urd) * (-4 + (3 + urd) * kappa) * log1_minus_a;
  double term6 = (-1 + urd) * (-2 + kappa + urd * kappa) * pow(log1_minus_a, 2);
  double term7 =
      (1 + urd) * log1_plus_a *
      (-4 - (-3 + urd) * kappa + (2 + (-1 + urd) * kappa) * log1_plus_a);

  double numerator2 = term4 + term5 - term6 + term7;

  /* Final result */
  double result =
      -(1 / (16 * pow(urd, 2))) * (numerator1_squared - 4 * urd * numerator2);
  return result;
}

double getLogDistBound(double urd, double kappa) { return log(1 + urd); }

double getGamma(int N, Precision prec, BoundType btype, double confidence) {
  double urd = computeUnitRoundOff(prec);
  double gamma, kappa;
  kappa = 0.0;  // kappa = 0 means that relative error is moded as uniform
                // distribution

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
    gamma = exp(t + (N * urd * urd) / (1.0 - urd)) - 1.0;
  } else if (btype == Bernstein) {
    /* double mu = (-2.0 * urd + (-1.0 + urd) * log(1.0 - urd) + */
    /*              (1.0 + urd) * log(1.0 + urd)) / */
    /*             (2.0 * urd); */
    /* double kappa = -1.0 + pow(urd, 2); */
    /* double c = log(1.0 + urd); */
    /* double var = */
    /*     (4 * pow(urd, 2) + kappa * (pow(log(1.0 - urd), 2) - */
    /*                                 2.0 * log(1.0 - urd) * log(1.0 + urd) +
     */
    /*                                 pow(log(1.0 + urd), 2))) / */
    /*     (4 * pow(urd, 2)); */
    double mu = getLogDistMean(urd, kappa);
    double var = getLogDistVar(urd, kappa);
    double c = getLogDistBound(urd, kappa);
    double log_term = log((1 - confidence) / 2.0);
    /* std::cout << prec << " " << kappa << " "<<   mu << " " << var << " " <<
     * std::endl; */
    double t = (1.0 / 3.0) * (-c * log_term + sqrt(pow(c * log_term, 2) -
                                                   18.0 * N * log_term * var));
    gamma = exp(t + N * abs(mu)) - 1.0;
  } else {
    std::invalid_argument("Invalid bound type");
  }
  return gamma;
}

void compareGamma(int N_lower, Precision prec, std::string filename,
                  int bit_shift, int max_shift, double confidence) {
  int N = N_lower;
  double gamma_deterministic, gamma_hoeffding, gamma_bernstein;
  const int width_int = 6;
  const int width_double = 15;
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
