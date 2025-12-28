#include "prob_model.hpp"

#include <cmath>
#include <stdexcept>

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
  stats.mean = mean;
  stats.var = var;
  stats.bound = bound;

  return stats;
}

// get the statistics of log(1+delta) random varianble
log1pdeltastats get_log1pdelta_stats(Precision prec, BoundModel bound_model,
                                     double beta_dist_alpha,
                                     double beta_dist_beta) {
  log1pdeltastats stats;
  if (bound_model == Uniform) {
    stats = compute_uniform_model_stats(prec);
  } else if (bound_model == Beta) {
    stats = compute_beta_model_stats(prec, beta_dist_alpha, beta_dist_beta);
  }

  return stats;
}
