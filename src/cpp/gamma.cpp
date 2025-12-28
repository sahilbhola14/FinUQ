#include "gamma.hpp"

#include <cmath>

#include "prob_model.hpp"
#include "utils.hpp"

double compute_determinsitic_gamma(int n, Precision prec) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* compute gamma value */
  double gamma = n * urd / (1.0 - n * urd);
  if (gamma > 1.0) {
    return 1.0;
  } else {
    return gamma;
  }
}

double compute_hoeffding_gamma(int n, Precision prec, float confidence) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* compute the lambda parameter */
  double lambda =
      (1.0 / (1.0 - urd)) * std::sqrt(2.0 * std::log(2.0 / (1.0 - confidence)));
  /* compute the bound */
  double c = urd / (1.0 - urd);
  /* compute the root of the quadratic */
  double t_plus = c * std::sqrt(2 * n * std::log(2.0 / (1.0 - confidence)));
  /* compute the gamma value */
  double gamma = std::exp(t_plus + (n * std::pow(urd, 2) / (1.0 - urd))) - 1.0;

  if (gamma > 1.0) {
    return 1.0;
  } else {
    return gamma;
  }
}

double compute_bernstein_gamma(int n, Precision prec, float confidence,
                               BoundModel bound_model = Uniform,
                               double beta_dist_alpha = 4.0,
                               double beta_dist_beta = 2.0) {
  /* utils */
  log1pdeltastats stats;
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* compute the statistics of log(1+delta) distribution; */
  stats = get_log1pdelta_stats(prec, Beta);
  /* stats = compute_uniform_model_stats(prec); */
  std::cout << stats.mean << std::endl;
  std::cout << stats.var << std::endl;
  std::cout << stats.bound << std::endl;
  /* stats = get_log1pdelta_stats(prec, bound_model); */

  /* /1* compute the lambda parameter *1/ */
  /* double lambda = (1.0 / (1.0 - urd)) * std::sqrt(2.0 *
   * std::log(2.0/(1.0-confidence))); */
  /* /1* compute the bound *1/ */
  /* double c = urd / (1.0 - urd); */
  /* /1* compute the root of the quadratic *1/ */
  /* double t_plus = c * std::sqrt(2*n*std::log(2.0/(1.0 - confidence))); */
  /* /1* compute the gamma value *1/ */
  /* double gamma = std::exp(t_plus + (n*std::pow(urd, 2)/(1.0-urd))) - 1.0; */

  /* if (gamma > 1.0){ */
  /*     return 1.0; */
  /* } else { */
  /*     return gamma; */
  /* } */
  return 0.0;
}

void compare_gamma(const gamma_config &cfg) {
  double gamma_det, gamma_mprea, gamma_vprea;
  /* determinisitic gamma */
  gamma_det = compute_determinsitic_gamma(10000000, cfg.prec);
  /* probabilistic gamma (mean informed) */
  gamma_mprea = compute_hoeffding_gamma(10000000, cfg.prec, cfg.confidence);
  /* probabilistic gamma (variance-informed) */
  gamma_vprea = compute_bernstein_gamma(10000000, cfg.prec, cfg.confidence);

  /* std::cout << gamma_det << std::endl; */
  /* std::cout << gamma_mprea << std::endl; */
}
