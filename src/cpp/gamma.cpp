
#include "gamma.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "prob_model.hpp"
#include "utils.hpp"

double compute_determinsitic_gamma(int n, Precision prec) {
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* compute gamma value */
  double gamma = (n * urd) / (1.0 - n * urd);
  if ((gamma > 1.0) || (n * urd > 1.0)) {
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
  double bound = urd / (1.0 - urd);
  /* compute the positive root of the quadratic */
  double t_plus = bound * std::sqrt(2 * n * std::log(2.0 / (1.0 - confidence)));
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
  /* assert statements */
  assert((confidence < 1.0) && "confidence must be < 1.0");

  /* utils */
  log1pdeltastats stats;
  double logc = std::log((1.0 - confidence) / 2.0);

  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  /* compute the statistics of log(1+delta) distribution; */
  stats =
      get_log1pdelta_stats(prec, bound_model, beta_dist_alpha, beta_dist_beta);
  /* compute the positive root of the quadratic */
  double a_coeff = 1.0;
  double b_coeff = (2.0 / 3.0) * stats.bound * logc;
  double c_coeff = 2.0 * n * stats.var * logc;
  double t_plus =
      (-b_coeff + std::sqrt(std::pow(b_coeff, 2.0) - 4.0 * a_coeff * c_coeff)) /
      (2.0 * a_coeff);
  /* compute the gamma value */
  double gamma = std::exp(t_plus + (n * std::abs(stats.mean))) - 1.0;
  if (gamma > 1.0) {
    return 1.0;
  } else {
    return gamma;
  }
}

/* gamma filename */
std::string make_gamma_filename(const gamma_config &cfg) {
  std::ostringstream ss;
  ss << "gamma_" << to_string(cfg.prec) << "_prec"
     << "_confidence_" << std::fixed << std::setprecision(3) << cfg.confidence
     << "_" << to_string(cfg.bound_model);

  if (cfg.bound_model == Beta) {
    ss << "_a_" << cfg.beta_dist_alpha << "_b_" << cfg.beta_dist_beta;
  }

  ss << ".csv";

  return ss.str();
}

/* get gamma value */
gamma_result get_gamma(const int n, const gamma_config &cfg) {
  /* initialization */
  double gamma_det, gamma_mprea, gamma_vprea;
  gamma_result result;
  /* number of arithmetic operators */
  result.n = n;
  /* determinisitic gamma */
  result.gamma_det = compute_determinsitic_gamma(n, cfg.prec);
  /* probabilistic gamma (mean informed model) */
  result.gamma_mprea = compute_hoeffding_gamma(n, cfg.prec, cfg.confidence);
  /* probabilistic gamma (variance-informed) */
  result.gamma_vprea =
      compute_bernstein_gamma(n, cfg.prec, cfg.confidence, cfg.bound_model,
                              cfg.beta_dist_alpha, cfg.beta_dist_beta);
  return result;
}

/* compare gamma */
void compare_gamma(const gamma_config &cfg, bool verbose) {
  /* initialization */
  double gamma_det, gamma_mprea, gamma_vprea;
  std::vector<int> n_values = {10,     100,     1000,     10000,
                               100000, 1000000, 10000000, 100000000};
  std::vector<gamma_result> results;
  results.reserve(n_values.size());

  /* check the mean sign (for variance-informed model) */
  check_mean_rounding_error_sign(cfg.prec, cfg.bound_model, cfg.beta_dist_alpha,
                                 cfg.beta_dist_beta);
  /* compute */
  for (int n : n_values) {
    gamma_result r;
    /* update vector size */
    r.n = n;
    /* determinisitic gamma */
    r.gamma_det = compute_determinsitic_gamma(n, cfg.prec);
    /* probabilistic gamma (mean informed model) */
    r.gamma_mprea = compute_hoeffding_gamma(n, cfg.prec, cfg.confidence);
    /* probabilistic gamma (variance-informed) */
    r.gamma_vprea =
        compute_bernstein_gamma(n, cfg.prec, cfg.confidence, cfg.bound_model,
                                cfg.beta_dist_alpha, cfg.beta_dist_beta);
    /* update the results */
    results.push_back(r);
  }

  /* print */
  if (verbose == true) {
    std::cout << std::left << std::setw(12) << "n" << std::setw(18)
              << "gamma_det" << std::setw(18) << "gamma_mprea" << std::setw(18)
              << "gamma_vprea"
              << "\n";

    std::cout << std::string(66, '-') << "\n";
    std::cout << std::scientific << std::setprecision(6);

    for (const auto &r : results) {
      std::cout << std::left << std::setw(12) << r.n << std::setw(18)
                << r.gamma_det << std::setw(18) << r.gamma_mprea
                << std::setw(18) << r.gamma_vprea << "\n";
    }
  }

  /* save */
  std::string filename = make_gamma_filename(cfg);
  write_gamma_results_csv(results, filename, verbose);
}
