#include "prob_model.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "rounding_error_model.cuh"
#include "utils.hpp"

// compute the statistics of log(1+delta) when delta is U(-urd, urd)
log1pdeltastats compute_uniform_model_stats(Precision prec) {
  /* initialization */
  log1pdeltastats stats;
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* utils */
  double logm = std::log(1.0 - urd);
  double logp = std::log(1.0 + urd);
  double kappa = -1.0 + std::pow(urd, 2.0);
  /* compute the mean */
  double mean =
      (-2.0 * urd + (-1.0 + urd) * logm + (1.0 + urd) * logp) / (2.0 * urd);
  /* compute the variance */
  double var = (4.0 * std::pow(urd, 2.0) + kappa * std::pow(logm, 2.0) -
                2.0 * kappa * logm * logp + kappa * std::pow(logp, 2.0)) /
               (4.0 * std::pow(urd, 2.0));
  /* compute the bounds */
  double bound = logp;

  /* create the statistics structure */
  stats.mean = mean;
  stats.var = var;
  stats.bound = bound;

  return stats;
}

/* compute the statistics of log(1+delta)~ log(1-urd) +
(log(1+urd)-log(1-urd))*Z where Z~Beta(alpha,beta) distribution*/
log1pdeltastats compute_beta_model_stats(Precision prec,
                                         double beta_dist_alpha = 4.0,
                                         double beta_dist_beta = 2.0) {
  /* initialization */
  log1pdeltastats stats;
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* utils */
  double L = std::log((1.0 + urd) / (1.0 - urd));
  double p = beta_dist_alpha / (beta_dist_alpha + beta_dist_beta);
  double logm = std::log(1.0 - urd);
  double logp = std::log(1.0 + urd);
  /* compute the mean */
  double mean = logm + L * p;
  /* compute the variance */
  double denominator = std::pow(beta_dist_alpha + beta_dist_beta, 2.0) *
                       (beta_dist_alpha + beta_dist_beta + 1.0);
  double var =
      std::pow(L, 2.0) * (beta_dist_alpha * beta_dist_beta) / denominator;
  /* compute the bounds */
  double bound = logp;

  /* create the statistics structure */
  stats.mean = static_cast<long double>(mean);
  stats.var = static_cast<long double>(var);
  stats.bound = static_cast<long double>(bound);

  return stats;
}

/* check the sign of mean of delta, given parameters of beta distribution */
void check_mean_rounding_error_sign(Precision prec, BoundModel bound_model,
                                    double beta_dist_alpha,
                                    double beta_dist_beta) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* utils */
  double L = std::log((1.0 + urd) / (1.0 - urd));
  double p = beta_dist_alpha / (beta_dist_alpha + beta_dist_beta);
  double logm = std::log(1.0 - urd);
  double c = -logm / L;
  /* conditon */
  double condition = c * beta_dist_beta / (1.0 - c);

  if (bound_model == Uniform) {
    std::cout << "Rounding error random variable mean is zero" << std::endl;
  } else if (bound_model == Beta) {
    if (beta_dist_alpha > condition) {
      std::cout << "Rounding error random variable mean is strictly positive. "
                   "E[delta] > 0"
                << std::endl;
    } else if (std::pow(beta_dist_alpha - condition, 2.0) < 1e-15) {
      std::cout
          << "Rounding error random variable mean is non-negative. E[delta] >=0"
          << std::endl;
    } else {
      std::cout << "Rounding error random variable mean can be negative"
                << std::endl;
    }
  } else {
    throw std::invalid_argument("bound_model must be Uniform or Beta");
  }
}

// get the statistics of log(1+delta) random variable
log1pdeltastats get_log1pdelta_stats(Precision prec, BoundModel bound_model,
                                     double beta_dist_alpha,
                                     double beta_dist_beta, bool verbose) {
  log1pdeltastats stats;
  std::string print_string;

  if (bound_model == Uniform) {
    print_string = "uniform distribution model";
    stats = compute_uniform_model_stats(prec);
  } else if (bound_model == Beta) {
    print_string = "beta distribution model";
    stats = compute_beta_model_stats(prec, beta_dist_alpha, beta_dist_beta);
  }

  if (verbose == true) {
    std::cout << "[" << print_string << "] "
              << "Mean: " << stats.mean << ", Variance: " << stats.var
              << ", Bound: " << stats.bound << "\n";
  }

  return stats;
}

/*
 * get individual bound confidence value (zeta)
 * @ param: arithmetic_operations   number of arithmetic operations (n)
 * @ param: total_confidence        Q value of the bounds
 * Returns:
 * zeta in Q(n,zeta) = 1 - n(1 - zeta)
 * 1 - zeta = (1 - Q) / n
 * zeta = 1 - ((1 - Q) / n)
 * */
long double compute_individual_bound_one_minus_zeta(const int number_of_bounds,
                                                    double total_confidence) {
  /* initialize */
  long double Q = total_confidence;
  long double n_bounds = number_of_bounds;
  /* compute 1 - zeta */
  long double one_minus_zeta = (1.0L - Q) / n_bounds;
  long double zeta = 1.0L - one_minus_zeta;
  return one_minus_zeta;
}
