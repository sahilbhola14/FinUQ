#include "dot_product.hpp"

#include <cuda.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "backward_error.hpp"
#include "distribution.hpp"
#include "dot_product_cuda.cuh"
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
  }
  std::cout << "Bound confidence: " << dot_product_cfg.gamma_cfg.confidence
            << std::endl;
}

/* dot product filename */
std::string make_dot_product_filename(const dot_product_config &cfg) {
  std::ostringstream ss;
  ss << "dot_product_" << to_string(cfg.prec) << "_prec"
     << "_distribution_" << to_string(cfg.dist) << "_bound_confidence_"
     << std::fixed << std::setprecision(3) << cfg.gamma_cfg.confidence
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
void run_dot_product_experiment_fixed_size(
    const int n, const dot_product_config &dot_product_cfg,
    backward_error_result &result) {
  /* initialize */
  std::vector<T> h_a(n), h_b(n);
  std::vector<double> h_a_true(n), h_b_true(n);
  std::vector<double> h_a_true_abs(n), h_b_true_abs(n);
  std::vector<double> backward_error(dot_product_cfg.num_experiments);
  T h_result;
  double h_result_true, h_result_true_abs;
  gamma_result backward_error_bound;
  vector_stats backward_error_stats;

  /* run the experiment */
  for (int i = 0; i < dot_product_cfg.num_experiments; i++) {
    /* sample the vector */
    sample_random_vector(h_a, dot_product_cfg.prec, dot_product_cfg.dist, 0,
                         i * 2 + 4);  // a vector
    sample_random_vector(h_b, dot_product_cfg.prec, dot_product_cfg.dist, 0,
                         i * 2 + 3);                       // b vector
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
        h_result, h_result_true, h_result_true_abs, &backward_error[i]);
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

/* run dot product backward error experiment */
void run_dot_product_experiment(const dot_product_config &dot_product_cfg) {
  /* intialization */
  /* std::vector<int> n_values = {10, 100, 1000, 10000, 100000, 1000000}; */
  std::vector<int> n_values = {10, 100};
  std::vector<backward_error_result> results(n_values.size());

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-') << " Dot product Config "
            << std::string(10, '-') << std::endl;
  print_dot_product_config(dot_product_cfg);
  std::cout << "Vector size: ";
  for (const auto &v : n_values) std::cout << v << ", ";
  std::cout << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* assert statements */
  assert(dot_product_cfg.prec == dot_product_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  /* run experiment for fixed vector size */
  for (size_t i = 0; i < n_values.size(); i++) {
    switch (dot_product_cfg.prec) {
      case Double:
        run_dot_product_experiment_fixed_size<double>(
            n_values[i], dot_product_cfg, results[i]);
        break;
      case Single:
        run_dot_product_experiment_fixed_size<float>(
            n_values[i], dot_product_cfg, results[i]);
        break;
      case Half:
        run_dot_product_experiment_fixed_size<half>(
            n_values[i], dot_product_cfg, results[i]);
        break;
      default:
        throw std::invalid_argument("invalid precision");
    }
  }

  /* save */
  std::string filename = make_dot_product_filename(dot_product_cfg);
  write_backward_error_results_csv(results, filename);
}
