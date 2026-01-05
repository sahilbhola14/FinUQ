#include "dot_product.hpp"

#include <cuda.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "backward_error.hpp"
#include "distribution.hpp"
#include "dot_product_cuda.cuh"
#include "forward_error.hpp"
#include "prob_model.hpp"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* print dot product config */
void print_dot_product_config(const dot_product_config &dot_product_cfg) {
  std::cout << "Compute precision: " << to_string(dot_product_cfg.prec)
            << std::endl;
  std::cout << "Random distribution: " << to_string(dot_product_cfg.dist)
            << std::endl;
  std::cout << "Number of experiments: " << dot_product_cfg.num_experiments
            << std::endl;
  std::cout << "Bound model: "
            << to_string(dot_product_cfg.gamma_cfg.bound_model) << std::endl;
  if (dot_product_cfg.gamma_cfg.bound_model == Beta) {
    std::cout << "Beta bound model alpha value: "
              << dot_product_cfg.gamma_cfg.beta_dist_alpha << std::endl;
    std::cout << "Beta bound model beta value: "
              << dot_product_cfg.gamma_cfg.beta_dist_beta << std::endl;
    check_mean_rounding_error_sign(dot_product_cfg.prec,
                                   dot_product_cfg.gamma_cfg.bound_model,
                                   dot_product_cfg.gamma_cfg.beta_dist_alpha,
                                   dot_product_cfg.gamma_cfg.beta_dist_beta);
  }
  std::cout << "Bound confidence: " << std::setprecision(4)
            << dot_product_cfg.gamma_cfg.confidence << std::endl;
}

/* dot product filename */
std::string make_dot_product_filename(const std::string prefix,
                                      const dot_product_config &cfg) {
  std::ostringstream ss;
  ss << prefix << "_dot_product_" << to_string(cfg.prec) << "_prec"
     << "_distribution_" << to_string(cfg.dist) << "_bound_confidence_"
     << std::fixed << std::setprecision(5) << cfg.gamma_cfg.confidence
     << "_bound_model_" << to_string(cfg.gamma_cfg.bound_model);

  if (cfg.gamma_cfg.bound_model == Beta) {
    ss << "_a_" << cfg.gamma_cfg.beta_dist_alpha << "_b_"
       << cfg.gamma_cfg.beta_dist_beta;
  }

  ss << ".csv";

  return ss.str();
}

/*
 * run dot product backward error experiment for fixed size
 */
template <typename T>
void run_dot_product_backward_error_experiment_fixed_size(
    const int n, const dot_product_config &dot_product_cfg,
    backward_error_result &result, const int seed = 42) {
  /* initialize */
  std::vector<T> h_a(n), h_b(n);
  std::vector<double> h_a_true(n), h_b_true(n);
  std::vector<double> h_a_true_abs(n), h_b_true_abs(n);
  std::vector<double> backward_error(dot_product_cfg.num_experiments);
  T h_result;
  double h_result_true, h_result_true_abs;
  gamma_result backward_error_bound;
  vector_stats backward_error_stats;
  /* random state */
  std::mt19937 gen(seed);

  /* run the experiment */
  for (int i = 0; i < dot_product_cfg.num_experiments; i++) {
    /* sample the vector */
    sample_random_vector(h_a, dot_product_cfg.prec, dot_product_cfg.dist,
                         gen);  // a vector
    sample_random_vector(h_b, dot_product_cfg.prec, dot_product_cfg.dist,
                         gen);                             // b vector
    convert_vector_to_double(h_a, h_a_true);               // a true vector
    convert_vector_to_double(h_b, h_b_true);               // b true vector
    convert_vector_to_absolute_double(h_a, h_a_true_abs);  // |a| true vector
    convert_vector_to_absolute_double(h_b, h_b_true_abs);  // |b| true vector
    /* run the dot product(s) */
    launch_sequential_dot_product_kernel<T>(n, h_a, h_b, &h_result,
                                            dot_product_cfg.prec);
    launch_sequential_dot_product_kernel<double>(n, h_a_true, h_b_true,
                                                 &h_result_true, Double);
    launch_sequential_dot_product_kernel<double>(n, h_a_true_abs, h_b_true_abs,
                                                 &h_result_true_abs, Double);
    /* compute the backward error */
    compute_sequential_dot_product_backward_error(
        static_cast<double>(h_result), h_result_true, h_result_true_abs,
        &backward_error[i]);
  }

  /* compute the backward error statistics */
  backward_error_stats = get_vector_stats(backward_error);

  /* compute the backward error bound */
  backward_error_bound = compute_sequential_dot_product_backward_error_bound(
      n, dot_product_cfg.gamma_cfg);

  /* update result */
  result.n = n;
  result.backward_error_min = backward_error_stats.min;
  result.backward_error_max = backward_error_stats.max;
  result.backward_error_mean = backward_error_stats.mean;
  result.backward_error_bound = backward_error_bound;
}

/*
 * run dot product forward error experiment for fixed size
 */
template <typename T>
void run_dot_product_forward_error_experiment_fixed_size(
    const int n, const dot_product_config &dot_product_cfg,
    forward_error_result &result, const int seed = 42) {
  /* initialize */
  std::vector<T> h_a(n), h_b(n);
  std::vector<double> h_a_true(n), h_b_true(n);
  std::vector<double> h_a_true_abs(n), h_b_true_abs(n);
  T h_result;
  double h_result_true, h_result_true_abs, h_result_model;
  gamma_result forward_error_bound;
  /* random state */
  std::mt19937 gen(seed);

  /* run the experiment */
  for (int i = 0; i < dot_product_cfg.num_experiments; i++) {
    /* sample the vector */
    sample_random_vector(h_a, dot_product_cfg.prec, dot_product_cfg.dist,
                         gen);  // a vector
    sample_random_vector(h_b, dot_product_cfg.prec, dot_product_cfg.dist,
                         gen);                             // b vector
    convert_vector_to_double(h_a, h_a_true);               // a true vector
    convert_vector_to_double(h_b, h_b_true);               // b true vector
    convert_vector_to_absolute_double(h_a, h_a_true_abs);  // |a| true vector
    convert_vector_to_absolute_double(h_b, h_b_true_abs);  // |b| true vector
    /* run the dot product(s) */
    launch_sequential_dot_product_kernel<T>(n, h_a, h_b, &h_result,
                                            dot_product_cfg.prec);
    launch_sequential_dot_product_kernel<double>(n, h_a_true, h_b_true,
                                                 &h_result_true, Double);
    launch_sequential_dot_product_kernel<double>(n, h_a_true_abs, h_b_true_abs,
                                                 &h_result_true_abs, Double);
    launch_sequential_dot_product_model_kernel(
        n, h_a_true, h_b_true, &h_result_model, dot_product_cfg.prec,
        dot_product_cfg.gamma_cfg);
    /* compute the forward error */
    compute_sequential_dot_product_forward_error(
        static_cast<double>(h_result), h_result_true, &result.forward_error[i]);
    compute_sequential_dot_product_forward_error(
        h_result_model, h_result_true, &result.forward_error_model[i]);
    /* compute the forward error bound */
    result.forward_error_bound.push_back(
        compute_sequential_dot_product_forward_error_bound(
            n, h_result_true, h_result_true_abs, dot_product_cfg.gamma_cfg));
  }
}

/* run dot product backward error experiment */
void run_dot_product_backward_error_experiment(
    const dot_product_config &dot_product_cfg) {
  /* intialization */
  std::vector<int> vector_sizes = {10,     100,    1000,   10000,
                                   100000, 500000, 1000000};
  std::vector<backward_error_result> results(vector_sizes.size());

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Dot product backward error analysis config "
            << std::string(10, '-') << std::endl;
  print_dot_product_config(dot_product_cfg);
  std::cout << "Vector sizes: ";
  for (const auto &v : vector_sizes) std::cout << v << ", ";
  std::cout << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* assert statements */
  assert(dot_product_cfg.prec == dot_product_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  /* run experiment for fixed vector size */
  for (size_t i = 0; i < vector_sizes.size(); i++) {
    switch (dot_product_cfg.prec) {
      case Double:
        run_dot_product_backward_error_experiment_fixed_size<double>(
            vector_sizes[i], dot_product_cfg, results[i]);
        break;
      case Single:
        run_dot_product_backward_error_experiment_fixed_size<float>(
            vector_sizes[i], dot_product_cfg, results[i]);
        break;
      case Half:
        run_dot_product_backward_error_experiment_fixed_size<half>(
            vector_sizes[i], dot_product_cfg, results[i]);
        break;
      default:
        throw std::invalid_argument("invalid precision");
    }
  }

  /* save */
  std::string filename =
      make_dot_product_filename("backward_error_result", dot_product_cfg);
  write_backward_error_results_csv(results, filename);
}

/* run dot product forward error experiment */
void run_dot_product_forward_error_experiment(
    const int vector_size, const dot_product_config &dot_product_cfg) {
  /* initialization */
  forward_error_result results;
  results.n = vector_size;
  results.forward_error.resize(dot_product_cfg.num_experiments);
  results.forward_error_model.resize(dot_product_cfg.num_experiments);
  results.forward_error_bound.reserve(dot_product_cfg.num_experiments);

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Dot product forward error analysis config "
            << std::string(10, '-') << std::endl;
  print_dot_product_config(dot_product_cfg);
  std::cout << "Vector size: " << vector_size << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* assert statements */
  assert(dot_product_cfg.prec == dot_product_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  switch (dot_product_cfg.prec) {
    case Double:
      run_dot_product_forward_error_experiment_fixed_size<double>(
          vector_size, dot_product_cfg, results);
      break;
    case Single:
      run_dot_product_forward_error_experiment_fixed_size<float>(
          vector_size, dot_product_cfg, results);
      break;
    case Half:
      run_dot_product_forward_error_experiment_fixed_size<half>(
          vector_size, dot_product_cfg, results);
      break;
    default:
      throw std::invalid_argument("invalid precision");
  }

  /* save */
  std::ostringstream ss;
  ss << "forward_error_result_vector_size_" << vector_size;
  std::string filename = make_dot_product_filename(ss.str(), dot_product_cfg);
  write_forward_error_results_csv(results, filename);
}
