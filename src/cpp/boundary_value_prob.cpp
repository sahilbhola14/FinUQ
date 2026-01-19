#include "boundary_value_prob.hpp"

#include <cassert>
#include <iomanip>
#include <random>

#include "backward_error.hpp"
#include "distribution.hpp"
#include "prob_model.hpp"
#include "utils.hpp"

/* print bvp config */
void print_bvp_config(const bvp_config &bvp_cfg) {
  std::cout << "Compute precision: " << to_string(bvp_cfg.prec) << std::endl;
  std::cout << "Bound model: " << to_string(bvp_cfg.gamma_cfg.bound_model)
            << std::endl;
  if (bvp_cfg.gamma_cfg.bound_model == Beta) {
    std::cout << "Beta bound model alpha value: "
              << bvp_cfg.gamma_cfg.beta_dist_alpha << std::endl;
    std::cout << "Beta bound model beta value: "
              << bvp_cfg.gamma_cfg.beta_dist_beta << std::endl;
    check_mean_rounding_error_sign(bvp_cfg.prec, bvp_cfg.gamma_cfg.bound_model,
                                   bvp_cfg.gamma_cfg.beta_dist_alpha,
                                   bvp_cfg.gamma_cfg.beta_dist_beta);
  }
  std::cout << "Bound confidence: " << std::setprecision(4)
            << bvp_cfg.gamma_cfg.confidence << std::endl;
}

bvp_parameters sample_bvp_parameters(const int num_samples, std::mt19937 &gen) {
  /* initialize */
  bvp_parameters bvp_params;
  bvp_params.theta_one.resize(num_samples);
  bvp_params.theta_two.resize(num_samples);
  /* sample theta_1 ~ U(0.1, 1.1) */
  sample_uniform_distribution(bvp_params.theta_one, 0.1, 1.1, gen);
  /* /1* sample theta_2 ~ U(1, 2) *1/ */
  sample_uniform_distribution(bvp_params.theta_two, 1.0, 2.0, gen);
  return bvp_params;
}

/* compute analytical state */
void compute_analytical_state(std::vector<double> &state,
                              const double theta_one, const double theta_two,
                              bool verbose) {
  const int M = state.size();
  const std::vector<double> x_full = make_linspace(0.0, 1.0, M);
  for (int i = 0; i < M; i++) {
    state[i] = -(50.0 * std::pow(theta_two, 2.0) *
                 (x_full[i] * std::log1p(theta_one) -
                  std::log1p(x_full[i] * theta_one))) /
               (theta_one * std::log1p(theta_one));
    if (verbose == true) {
      printf("state at x: %f is %f\n", x_full[i], state[i]);
    }
  }
}

/* compute analytical state integral
 * compute p(\theta_1, \theta_2) = \int_x u dx.
 * */
void compute_analytical_state_integral(double &integral, const double theta_one,
                                       const double theta_two) {
  double numerator =
      25.0 * std::pow(theta_two, 2.0) *
      (-2.0 * theta_one + (2.0 + theta_one) * std::log1p(theta_one));
  double denominator = std::pow(theta_one, 2.0) * std::log1p(theta_one);
  integral = numerator / denominator;
}

/* compute analytical qoi using Monte Carlo integration */
void compute_analytical_qoi(const int num_samples, bool verbose,
                            const int seed) {
  /* initialize */
  double state_integral, mean;
  /* random generator */
  std::mt19937 gen(seed);
  /* sample the parameters */
  bvp_parameters bvp_params = sample_bvp_parameters(num_samples, gen);
  /* compute the state integral */
  for (int i = 0; i < num_samples; i++) {
    compute_analytical_state_integral(state_integral, bvp_params.theta_one[i],
                                      bvp_params.theta_two[i]);
    if (i == 0) {
      mean = state_integral;
    } else {
      mean = mean + (state_integral - mean) / i;
    }
  }
  /* print */
  if (verbose == true)
    printf("Analytical QoI (q): %.3e (using %d MCMC samples)\n", mean,
           num_samples);
}

/* template <typename T> */
/* void compute_the_diagonals( */
/*     std::vector<T> &sub_diag, */
/*     std::vector<T> &main_diag, */
/*     std::vector<T> &super_diag, */
/*     const double theta_one, */
/*     const double theta_two, */
/*     ){ */
/*   printf("dhfgl;ad"); */
/* } */

/* run bvp backward error experiment for fixed interval*/
template <typename T>
void run_ode_backward_error_experiment_fixed_interval(
    const bvp_config &bvp_cfg, const bvp_parameters &bvp_params,
    const int num_intervals, backward_error_result &result) {
  /* initialize */
  const int Ns = num_intervals - 1;  // number of states to solve
  /* const int num_samples = bvp_parameters.theta_one.size(); */
  std::vector<T> sub_diag(Ns), main_diag(Ns), super_diag(Ns);
  printf("dhfjkld");

  /* /1* run the experiment *1/ */
  /* for (int i=0; i < num_samples; i++){ */
  /*   /1* compute the diagonals *1/ */
  /*   compute_the_diagonals<T>( */
  /*       sub_diag, */
  /*       main_diag, */
  /*       super_diag, */
  /*       bvp_params.theta_one[i], */
  /*       bvp_params.theta_two[i], */
  /*       ); */

  /* } */
}

/* run bvp backward error experiment */
void run_ode_backward_error_experiment(const bvp_config &bvp_cfg,
                                       const int num_samples, const int seed) {
  /* initialize */
  std::vector<int> num_intervals = {4, 8, 16, 32, 64, 128};
  std::vector<backward_error_result> results(num_intervals.size());
  /* random generator */
  std::mt19937 gen(seed);

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-') << " ODE backward error analysis config "
            << std::string(10, '-') << std::endl;
  print_bvp_config(bvp_cfg);
  std::cout << "Num intervals: ";
  for (const auto &v : num_intervals) std::cout << v << ", ";
  std::cout << std::endl;
  std::cout << "Num parameters: " << num_samples << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* assert statements */
  assert(bvp_cfg.prec == bvp_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  /* sample the parameters */
  bvp_parameters bvp_params = sample_bvp_parameters(num_samples, gen);

  bvp_params.theta_one[0] = 1.0;  // for testing
  bvp_params.theta_two[0] = 1.0;  // for testing
  /* run experiment for fixed interval */
  run_ode_backward_error_experiment_fixed_interval<double>(
      bvp_cfg, bvp_params, num_intervals[0], results[0]);
}
