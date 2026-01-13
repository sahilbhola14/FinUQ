
#include "gamma.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "prob_model.hpp"
#include "rounding_error_model.cuh"
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

long double compute_hoeffding_gamma(int n, Precision prec,
                                    const long double one_minus_zeta) {
  /* initialization */
  long double nL = static_cast<long double>(n);
  /* assert statements */
  assert((one_minus_zeta < 1.0L) && "one minus zeta must be < 1.0");
  /* compute the unit roundoff */
  double urd = compute_unit_roundoff(prec);
  long double urdL = static_cast<long double>(urd);

  /* compute the lambda parameter */
  long double lambda =
      (1.0L / (1.0L - urdL)) *
      std::sqrt(2.0L * (std::log(2.0L) - std::log(one_minus_zeta)));

  /* compute the bound */
  long double bound = urdL / (1.0L - urdL);
  /* compute the positive root of the quadratic */
  long double t_plus =
      bound *
      std::sqrt(2.0L * nL * (std::log(2.0L) - std::log(one_minus_zeta)));
  /* compute the gamma value */
  long double gamma =
      std::exp(t_plus + (nL * std::pow(urdL, 2.0L) / (1.0L - urdL))) - 1.0L;

  if (gamma > 1.0L) {
    return 1.0L;
  } else {
    return gamma;
  }
}

double compute_bernstein_gamma(int n, Precision prec,
                               const long double one_minus_zeta,
                               BoundModel bound_model = Uniform,
                               double beta_dist_alpha = 4.0,
                               double beta_dist_beta = 2.0) {
  /* initialization */
  long double nL = static_cast<long double>(n);
  /* assert statements */
  assert((one_minus_zeta < 1.0L) && "one minus zeta must be < 1.0");

  /* utils */
  log1pdeltastats stats;
  long double logc = std::log(one_minus_zeta) - std::log(2.0L);

  /* compute the statistics of log(1+delta) distribution; */
  stats =
      get_log1pdelta_stats(prec, bound_model, beta_dist_alpha, beta_dist_beta);
  /* compute the positive root of the quadratic */
  long double a_coeff = 1.0L;
  long double b_coeff = (2.0L / 3.0L) * stats.bound * logc;
  long double c_coeff = 2.0L * nL * stats.var * logc;
  long double t_plus = (-b_coeff + std::sqrt(std::pow(b_coeff, 2.0L) -
                                             4.0L * a_coeff * c_coeff)) /
                       (2.0L * a_coeff);
  /* compute the gamma value */
  long double gamma = std::exp(t_plus + (nL * std::abs(stats.mean))) - 1.0L;
  if (gamma > 1.0L) {
    return 1.0L;
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
gamma_result get_gamma(const int n, const gamma_config &gamma_cfg,
                       const long double one_minus_zeta) {
  /* initialization */
  double gamma_det, gamma_mprea, gamma_vprea;
  gamma_result result;
  /* assert statements */
  assert(one_minus_zeta > 0.0L &&
         "Invalid one minus zeta must be strictly positive.\n");
  /* number of arithmetic operators */
  result.n = n;
  /* determinisitic gamma */
  result.gamma_det = compute_determinsitic_gamma(n, gamma_cfg.prec);
  /* probabilistic gamma (mean informed model) */
  result.gamma_mprea =
      compute_hoeffding_gamma(n, gamma_cfg.prec, one_minus_zeta);
  /* probabilistic gamma (variance-informed) */
  result.gamma_vprea = compute_bernstein_gamma(
      n, gamma_cfg.prec, one_minus_zeta, gamma_cfg.bound_model,
      gamma_cfg.beta_dist_alpha, gamma_cfg.beta_dist_beta);
  return result;
}

/* compare gamma */
void compare_gamma(const gamma_config &gamma_cfg, bool verbose) {
  /* initialization */
  double gamma_det, gamma_mprea, gamma_vprea;
  /* std::vector<int> n_values = {1, 5, 10,     100,     1000,     10000, */
  /*                              100000, 1000000, 10000000, 50000000,
   * 100000000}; */
  std::vector<int> n_values = make_logspace(1, 100000000, 100);
  std::vector<gamma_result> results;
  results.reserve(n_values.size());

  /* compute one one_minus_zeta with single bound to be satisfied*/
  long double one_minus_zeta =
      compute_individual_bound_one_minus_zeta(1, gamma_cfg.confidence);

  /* check the mean sign (for variance-informed model) */
  check_mean_rounding_error_sign(gamma_cfg.prec, gamma_cfg.bound_model,
                                 gamma_cfg.beta_dist_alpha,
                                 gamma_cfg.beta_dist_beta);

  /* compute */
  for (int n : n_values) {
    gamma_result r;
    /* update vector size */
    r.n = n;
    /* determinisitic gamma */
    r.gamma_det = compute_determinsitic_gamma(n, gamma_cfg.prec);
    /* probabilistic gamma (mean informed model) */
    r.gamma_mprea = compute_hoeffding_gamma(n, gamma_cfg.prec, one_minus_zeta);
    /* probabilistic gamma (variance-informed) */
    r.gamma_vprea = compute_bernstein_gamma(
        n, gamma_cfg.prec, one_minus_zeta, gamma_cfg.bound_model,
        gamma_cfg.beta_dist_alpha, gamma_cfg.beta_dist_beta);
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
  std::string filename = make_gamma_filename(gamma_cfg);
  write_gamma_results_csv(results, filename, verbose);
}

/* experiments */
void run_all_compare_gamma_experiments(Precision prec) {
  /* configuration */
  gamma_config gamma_cfg;
  gamma_cfg.prec = prec;
  /* vary confidence for uniform model*/
  gamma_cfg.bound_model = Uniform;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 1.9, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 1.9;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 1.95, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 1.95;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 1.97, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 1.97;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 2.0, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 2.0;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
}
