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

/* run backward error experiment for solving the ode for fixed interval*/
template <typename T>
void run_backward_error_ode_sol_experiment_fixed_interval(
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
    if (i % 10 == 0) {
      printf(
          "Running backward error experiment : %d/%d for num intervals: %d\n",
          i + 1, num_samples, num_intervals);
    }

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

  /* compute the backward error bound */
  backward_error_bound =
      compute_ode_backward_error_bound(num_intervals, bvp_cfg.gamma_cfg);

  /* update result */
  result.n = num_intervals;
  result.backward_error_min = backward_error_stats.min;
  result.backward_error_max = backward_error_stats.max;
  result.backward_error_mean = backward_error_stats.mean;
  result.backward_error_bound = backward_error_bound;
}

/* run forward error experiment for solving the ode for fixed interval*/
template <typename T>
void run_forward_error_qoi_experiment_fixed_interval(
    const bvp_config &bvp_cfg, const bvp_parameters &bvp_params,
    const int num_intervals, bvp_forward_error_result &result) {
  /* initialize */
  const int Ns = num_intervals - 1;  // number of states to solve
  assert(bvp_params.theta_one.size() == bvp_params.theta_two.size() &&
         "inconsistent number of params");
  const int num_samples = bvp_params.theta_one.size();
  std::vector<gamma_result> forward_error_bound_state_integral;
  forward_error_bound_state_integral.reserve(num_samples);

  std::vector<T> h_sub_diag(Ns), h_main_diag(Ns), h_super_diag(Ns), h_rhs(Ns),
      h_state(Ns), h_state_integral(num_samples);

  T h_qoi;

  std::vector<double> h_sub_diag_true(Ns), h_main_diag_true(Ns),
      h_super_diag_true(Ns), h_rhs_true(Ns), h_state_true(Ns),
      h_state_integral_true(num_samples);

  double h_qoi_true;

  std::vector<double> h_state_model(Ns), h_state_integral_model(num_samples);
  double h_qoi_model;

  /* run the experiment */
  for (int i = 0; i < num_samples; i++) {
    if (i % 10 == 0) {
      printf("Running forward error experiment : %d/%d for num intervals: %d\n",
             i + 1, num_samples, num_intervals);
    }

    /* compute the diagonals */
    compute_the_diagonals<T>(h_sub_diag, h_main_diag, h_super_diag,
                             bvp_params.theta_one[i], bvp_params.theta_two[i]);
    convert_vector_to_double(h_sub_diag, h_sub_diag_true);
    convert_vector_to_double(h_main_diag, h_main_diag_true);
    convert_vector_to_double(h_super_diag, h_super_diag_true);
    /* compute the rhs */
    compute_rhs<T>(h_rhs, num_intervals, bvp_params.theta_one[i],
                   bvp_params.theta_two[i]);
    convert_vector_to_double(h_rhs, h_rhs_true);
    /* compute the solution state(s) using Thomas algorithm */
    launch_thomas_algorithm_kernel<T>(num_intervals, h_sub_diag, h_main_diag,
                                      h_super_diag, h_rhs, h_state,
                                      bvp_cfg.prec);
    launch_thomas_algorithm_kernel<double>(num_intervals, h_sub_diag_true,
                                           h_main_diag_true, h_super_diag_true,
                                           h_rhs_true, h_state_true, Double);
    launch_thomas_algorithm_model_kernel(
        num_intervals, h_sub_diag_true, h_main_diag_true, h_super_diag_true,
        h_rhs_true, h_state_model, bvp_cfg.prec, bvp_cfg.gamma_cfg, i);

    /* integrate the state(s) using Reimann integration */
    launch_state_integral_kernel<T>(num_intervals, h_state, h_state_integral[i],
                                    bvp_cfg.prec);
    launch_state_integral_kernel<double>(num_intervals, h_state_true,
                                         h_state_integral_true[i], Double);
    launch_state_integral_model_kernel(num_intervals, h_state_model,
                                       h_state_integral_model[i], bvp_cfg.prec,
                                       bvp_cfg.gamma_cfg, i);

    /* compute the forward error bounds for the state integral */
    forward_error_bound_state_integral.push_back(
        compute_bvp_state_integral_forward_error_bound(
            num_intervals, num_samples, h_sub_diag, h_main_diag, h_super_diag,
            h_state, bvp_cfg.gamma_cfg, false));

    // printf("state integral error : %.3e \n",
    // std::abs(static_cast<double>(h_state_integral[i]) -
    // h_state_integral_true[i]));
  }

  /* compute the qoi */
  launch_monte_carlo_expectation_kernel<T>(h_state_integral, h_qoi,
                                           bvp_cfg.prec);
  launch_monte_carlo_expectation_kernel<double>(h_state_integral_true,
                                                h_qoi_true, Double);
  launch_monte_carlo_expectation_model_kernel(
      h_state_integral_model, h_qoi_model, bvp_cfg.prec, bvp_cfg.gamma_cfg,
      num_samples + 1);

  /* compute the forward error in qoi */
  compute_bvp_qoi_forward_error(static_cast<double>(h_qoi), h_qoi_true,
                                &result.qoi_forward_error, true);

  compute_bvp_qoi_forward_error(static_cast<double>(h_qoi_model), h_qoi_true,
                                &result.qoi_forward_error_model, false);

  /* compute the forward error bound for the QoI */
  gamma_result forward_error_bound_qoi = compute_bvp_qoi_forward_error_bound(
      num_intervals, num_samples, h_state_integral,
      forward_error_bound_state_integral, bvp_cfg.gamma_cfg, true);

  // update results with bounds
  result.forward_error_bound = forward_error_bound_qoi;
}

/* run backward error experiment for solving the ode*/
void run_backward_error_ode_sol_experiment(const bvp_config &bvp_cfg,
                                           const int num_samples,
                                           const int seed = 42) {
  /* initialize */
  std::vector<int> num_intervals = {16,  32,   64,   128, 256,
                                    512, 1024, 2048, 4069};
  std::vector<backward_error_result> results(num_intervals.size());
  /* random generator */
  std::mt19937 gen(seed);

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " ODE solution backward error analysis config "
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
        run_backward_error_ode_sol_experiment_fixed_interval<double>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      case Single: {
        run_backward_error_ode_sol_experiment_fixed_interval<float>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      case Half: {
        run_backward_error_ode_sol_experiment_fixed_interval<half>(
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

/* run forward error experiment for solving the ode*/
void run_forward_error_qoi_experiment(const bvp_config &bvp_cfg,
                                      const int num_samples,
                                      const int seed = 42) {
  /* initialize */
  // std::vector<int> num_intervals = {16,  32,   64,   128, 256, 512, 1024,
  // 2048, 4069};
  std::vector<int> num_intervals = {128};
  // std::vector<int> num_intervals = {128};
  std::vector<bvp_forward_error_result> results(num_intervals.size());

  /* random generator */
  std::mt19937 gen(seed);

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " ODE solution forward error analysis config "
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

  // for (int i = 0; i < num_samples; i++){
  //   bvp_params.theta_one[i] = 1.0;
  //   bvp_params.theta_two[i] = 1.0;
  // }

  /* run experiment for fixed interval */
  for (size_t i = 0; i < num_intervals.size(); i++) {
    /* update result with characteristic dimension */
    results[i].n = num_intervals[i];
    /* run */
    switch (bvp_cfg.prec) {
      case Double: {
        run_forward_error_qoi_experiment_fixed_interval<double>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      case Single: {
        run_forward_error_qoi_experiment_fixed_interval<float>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      case Half: {
        run_forward_error_qoi_experiment_fixed_interval<half>(
            bvp_cfg, bvp_params, num_intervals[i], results[i]);
        break;
      }
      default: {
        throw std::invalid_argument("invalid precision");
      }
    }
  }

  /* save */
  std::ostringstream ss;
  ss << "forward_error_result_num_samples_" << num_samples;
  std::string filename = make_bvp_filename(ss.str(), bvp_cfg);
  write_bvp_forward_error_results_csv(results, filename);
}

namespace bvp {

// backward error in solving the tri-diagonal system
void run_all_backward_error_ode_sol_experiments(Precision prec,
                                                const int num_samples = 10000) {
  /* configuration */
  bvp_config bvp_cfg;
  bvp_cfg.prec = prec;
  bvp_cfg.prec = prec;                  // sampling precision
  bvp_cfg.gamma_cfg.prec = prec;        // bound precision
  bvp_cfg.gamma_cfg.confidence = 0.99;  // overall confidence
  // beta shape parameter
  bvp_cfg.gamma_cfg.beta_dist_beta = 2.0;
  // alpha shape parameter
  std::vector<double> beta_dist_alpha_vals = {1.6, 1.7, 1.8, 1.9, 2.0};

  // run uniform model
  bvp_cfg.gamma_cfg.bound_model = Uniform;
  run_backward_error_ode_sol_experiment(bvp_cfg, num_samples);

  // run beta model
  bvp_cfg.gamma_cfg.bound_model = Beta;
  for (auto &alpha : beta_dist_alpha_vals) {
    bvp_cfg.gamma_cfg.beta_dist_alpha = alpha;
    run_backward_error_ode_sol_experiment(bvp_cfg, num_samples);
  }
}

// forward error in solving the tri-diagonal system
void run_all_forward_error_qoi_experiments(Precision prec,
                                           const int num_samples = 1) {
  /* configuration */
  bvp_config bvp_cfg;
  bvp_cfg.prec = prec;
  bvp_cfg.prec = prec;                  // sampling precision
  bvp_cfg.gamma_cfg.prec = prec;        // bound precision
  bvp_cfg.gamma_cfg.confidence = 0.99;  // overall confidence
  // beta shape parameter
  bvp_cfg.gamma_cfg.beta_dist_beta = 2.0;
  // alpha shape parameter
  std::vector<double> beta_dist_alpha_vals = {1.3, 1.4, 1.5, 1.6, 1.7};

  // run uniform model
  bvp_cfg.gamma_cfg.bound_model = Uniform;
  run_forward_error_qoi_experiment(bvp_cfg, num_samples);

  // // run beta model
  // bvp_cfg.gamma_cfg.bound_model = Beta;
  // for (auto &alpha : beta_dist_alpha_vals) {
  //   bvp_cfg.gamma_cfg.beta_dist_alpha = alpha;
  //   run_forward_error_qoi_experiment(bvp_cfg, num_samples);
  // }
}

}  // namespace bvp

/* run all experiments */
void run_all_ode_experiments(Precision prec) {
  // backward error in solving the tri-diagonal system
  /* bvp::run_all_backward_error_ode_sol_experiments(prec); */
  // forward error in obtainig the QoI
  bvp::run_all_forward_error_qoi_experiments(prec, 5000);
}
