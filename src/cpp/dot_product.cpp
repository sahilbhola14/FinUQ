#include "dot_product.hpp"

#include <cuda.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "backward_error.hpp"
#include "distribution.hpp"
#include "dot_product_cuda.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* run dot product experiment for fixed size */
template <typename T>
void run_dot_product_experiment_fixed_size(
    const int n, const dot_product_config &dot_product_cfg) {
  /* initialize */
  std::vector<T> h_a(n), h_b(n);
  std::vector<double> h_a_true(n), h_b_true(n);
  std::vector<double> h_a_true_abs(n), h_b_true_abs(n);
  std::vector<double> backward_error;
  backward_error.reserve(dot_product_cfg.num_experiments);
  T h_result;
  double h_result_true, h_result_true_abs;
  double backward_error_bound;

  /* run the experiment */
  for (int i = 0; i < dot_product_cfg.num_experiments; i++) {
    /* sample the vector */
    sample_random_vector(h_a, dot_product_cfg.prec,
                         dot_product_cfg.dist);  // a vector
    sample_random_vector(h_b, dot_product_cfg.prec,
                         dot_product_cfg.dist);            // b vector
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

  /* compute the backward error bound */
  gamma_result backward_error_bound =
      compute_sequential_dot_product_backward_error_bound(
          n, dot_product_cfg.gamma_cfg, true);
}

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

void run_dot_product_experiment(const dot_product_config &dot_product_cfg) {
  /* intialization */
  std::vector<int> n_values = {10, 100, 1000, 10000, 100000, 1000000};
  std::vector<dot_product_result> results;
  results.reserve(n_values.size());

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
  int run_n = 1;
  switch (dot_product_cfg.prec) {
    case Double:
      run_dot_product_experiment_fixed_size<double>(n_values[run_n],
                                                    dot_product_cfg);
      break;
    case Single:
      run_dot_product_experiment_fixed_size<float>(n_values[run_n],
                                                   dot_product_cfg);
      break;
    case Half:
      run_dot_product_experiment_fixed_size<half>(n_values[run_n],
                                                  dot_product_cfg);
      break;
    default:
      throw std::invalid_argument("invalid precision");
  }
}
