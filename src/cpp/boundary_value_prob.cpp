#include "boundary_value_prob.hpp"

#include <cassert>
#include <iomanip>
#include <random>

#include "backward_error.hpp"
#include "boundary_value_prob_cuda.cuh"
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

/* BVP filename */
std::string make_bvp_filename(const std::string prefix, const bvp_config &cfg) {
  std::ostringstream ss;
  ss << prefix << "_bvp_" << to_string(cfg.prec) << "_prec"
     << "_bound_confidence_" << std::fixed << std::setprecision(5)
     << cfg.gamma_cfg.confidence << "_bound_model_"
     << to_string(cfg.gamma_cfg.bound_model);

  if (cfg.gamma_cfg.bound_model == Beta) {
    ss << "_a_" << cfg.gamma_cfg.beta_dist_alpha << "_b_"
       << cfg.gamma_cfg.beta_dist_beta;
  }

  ss << ".csv";

  return ss.str();
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

/* compute the diagonals */
template <typename T>
void compute_the_diagonals(std::vector<T> &sub_diag, std::vector<T> &main_diag,
                           std::vector<T> &super_diag, const double theta_one,
                           const double theta_two) {
  /* initialize */
  const int num_intervals = sub_diag.size() + 1;
  const T delta_x = static_cast<T>(1.0 / num_intervals);
  const T delta_x_sq = delta_x * delta_x;
  const T one_half = static_cast<T>(0.5);
  const T one = static_cast<T>(1.0);
  const T theta_one_conv = static_cast<T>(theta_one);

  /* compute constant vector and diagonals */
  for (int i = 1; i <= num_intervals; i++) {
    T x = static_cast<T>(i) * delta_x;
    T constant_val = one + theta_one_conv * (x - one_half * delta_x);
    if (i < num_intervals) {
      sub_diag[i - 1] = constant_val;
    }
    if (i > 1) {
      super_diag[i - 2] = constant_val;
    }
  }
  /* compute the main diagonal */
  for (int i = 0; i < static_cast<int>(sub_diag.size()); ++i) {
    main_diag[i] = -(sub_diag[i] + super_diag[i]);
  }
}

/* compute the rhs */
template <typename T>
void compute_rhs(std::vector<T> &rhs, const int num_intervals,
                 const double theta_one, const double theta_two) {
  /* initialize */
  const T delta_x = static_cast<T>(1.0 / num_intervals);
  const T delta_x_sq = delta_x * delta_x;
  const T theta_two_conv = static_cast<T>(theta_two);
  const T fifty = static_cast<T>(50.0);
  const T coeff = -fifty * theta_two_conv * theta_two_conv * delta_x_sq;
  /* fill rhs vector */
  for (int i = 0; i < num_intervals - 1; i++) {
    rhs[i] = coeff;
  }
}

/* run bvp backward error experiment for fixed interval*/
template <typename T>
void run_ode_backward_error_experiment_fixed_interval(
    const bvp_config &bvp_cfg, const bvp_parameters &bvp_params,
    const int num_intervals, backward_error_result &result) {
  /* initialize */
  const int Ns = num_intervals - 1;  // number of states to solve
  assert(bvp_params.theta_one.size() == bvp_params.theta_two.size() &&
         "inconsistent number of params");
  const int num_samples = bvp_params.theta_one.size();
  std::vector<double> backward_error(num_samples);
  vector_stats backward_error_stats;
  gamma_result backward_error_bound;

  std::vector<T> h_sub_diag(Ns), h_main_diag(Ns), h_super_diag(Ns), h_rhs(Ns),
      h_state(Ns);

  /* run the experiment */
  for (int i = 0; i < num_samples; i++) {
    /* compute the diagonals */
    compute_the_diagonals<T>(h_sub_diag, h_main_diag, h_super_diag,
                             bvp_params.theta_one[i], bvp_params.theta_two[i]);
    /* compute the rhs */
    compute_rhs<T>(h_rhs, num_intervals, bvp_params.theta_one[i],
                   bvp_params.theta_two[i]);
    /* compute the solution state */
    launch_thomas_algorithm_kernel<T>(num_intervals, h_sub_diag, h_main_diag,
                                      h_super_diag, h_rhs, h_state,
                                      bvp_cfg.prec);
    /* compute the backward error */
    compute_ode_backward_error(num_intervals, h_sub_diag, h_main_diag,
                               h_super_diag, h_rhs, h_state, &backward_error[i],
                               bvp_cfg.prec);
  }

  /* compute the backward error statistics */
  backward_error_stats = get_vector_stats(backward_error);
  printf("%.3e\n", backward_error_stats.mean);

  /* compute the backward error bound */
  backward_error_bound =
      compute_ode_backward_error_bound(num_intervals, bvp_cfg.gamma_cfg, true);

  /* update result */
  result.n = num_intervals;
  result.backward_error_min = backward_error_stats.min;
  result.backward_error_max = backward_error_stats.max;
  result.backward_error_mean = backward_error_stats.mean;
  result.backward_error_bound = backward_error_bound;
}

/* run bvp backward error experiment */
void run_ode_backward_error_experiment(const bvp_config &bvp_cfg,
                                       const int num_samples, const int seed) {
  /* initialize */
  std::vector<int> num_intervals = {4,   8,   16,   32,   64,  128,
                                    256, 512, 1024, 2048, 4069};
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

  /* run experiment for fixed interval */
  for (size_t i = 0; i < num_intervals.size(); i++) {
    switch (bvp_cfg.prec) {
      case Double: {
        run_ode_backward_error_experiment_fixed_interval<double>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      case Single: {
        run_ode_backward_error_experiment_fixed_interval<float>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      case Half: {
        run_ode_backward_error_experiment_fixed_interval<half>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      default: {
        throw std::invalid_argument("invalid precision");
      }
    }
  }

  /* save */
  std::string filename = make_bvp_filename("backward_error_result", bvp_cfg);
  write_backward_error_results_csv(results, filename);
}
