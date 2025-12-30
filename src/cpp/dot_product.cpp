#include "dot_product.hpp"

#include <cuda.h>
#include <cuda_fp16.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "distribution.hpp"
#include "dot_product_cuda.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* run dot product experiment for fixed size */
template <typename T>
void run_dot_product_experiment_fixed_size(const int n, Precision prec,
                                           Distribution dist,
                                           const int num_experiments = 100) {
  /* initialize */
  std::vector<T> h_a(n), h_b(n);
  std::vector<double> h_a_true(n), h_b_true(n);
  T h_result;
  double h_result_true;

  /* run the experiment */
  for (int i = 0; i < num_experiments; i++) {
    /* sample the vector */
    sample_random_vector(h_a, prec, dist);
    sample_random_vector(h_b, prec, dist);
    convert_vector_to_double(h_a, h_a_true);
    convert_vector_to_double(h_b, h_b_true);
    /* run the dot product(s) */
    launch_sequential_dot_product_kernel<T>(n, h_a, h_b, &h_result, prec);
    launch_sequential_dot_product_kernel<double>(n, h_a_true, h_b_true,
                                                 &h_result_true, prec);
    std::cout << static_cast<double>(h_result) << std::endl;
    std::cout << static_cast<double>(h_result_true) << std::endl;
  }
}

void run_dot_product_experiment(Precision prec, Distribution dist,
                                const int num_experiments) {
  /* intialization */
  std::vector<int> n_values = {10, 100, 1000, 10000, 100000, 1000000};
  std::vector<dot_product_result> results;
  results.reserve(n_values.size());
  std::cout << std::string(50, '=') << std::endl;
  std::cout << "Dot product Experiment" << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* run experiment for fixed vector size */
  switch (prec) {
    case Double:
      run_dot_product_experiment_fixed_size<double>(n_values.front(), prec,
                                                    dist, num_experiments);
      break;
    case Single:
      run_dot_product_experiment_fixed_size<float>(n_values.front(), prec, dist,
                                                   num_experiments);
      break;
    case Half:
      run_dot_product_experiment_fixed_size<half>(n_values.front(), prec, dist,
                                                  num_experiments);
      break;
    default:
      throw std::invalid_argument("invalid precision");
  }
}
