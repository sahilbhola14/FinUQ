#include "matrix_vector.hpp"

#include <cuda.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

#include "distribution.hpp"
#include "matrix_vector_cuda.cuh"
#include "prob_model.hpp"
#include "utils.hpp"

/* print matvec product config */
void print_matvec_product_config(
    const matvec_product_config &matvec_product_cfg) {
  std::cout << "Compute precision: " << to_string(matvec_product_cfg.prec)
            << std::endl;
  std::cout << "Random distribution: " << to_string(matvec_product_cfg.dist)
            << std::endl;
  std::cout << "Number of experiments: " << matvec_product_cfg.num_experiments
            << std::endl;
  std::cout << "Bound model: "
            << to_string(matvec_product_cfg.gamma_cfg.bound_model) << std::endl;
  if (matvec_product_cfg.gamma_cfg.bound_model == Beta) {
    std::cout << "Beta bound model alpha value: "
              << matvec_product_cfg.gamma_cfg.beta_dist_alpha << std::endl;
    std::cout << "Beta bound model beta value: "
              << matvec_product_cfg.gamma_cfg.beta_dist_beta << std::endl;
    check_mean_rounding_error_sign(matvec_product_cfg.prec,
                                   matvec_product_cfg.gamma_cfg.bound_model,
                                   matvec_product_cfg.gamma_cfg.beta_dist_alpha,
                                   matvec_product_cfg.gamma_cfg.beta_dist_beta);
  }
  std::cout << "Bound confidence: " << std::setprecision(4)
            << matvec_product_cfg.gamma_cfg.confidence << std::endl;
}

/*
 * run matrix vector product backward error experiment for fixed size
 */
template <typename T>
void run_matvec_product_backward_error_experiment_given_matrix(
    const Matrix<T> &h_matrix, const matvec_product_config &matvec_product_cfg,
    backward_error_result &result, const int seed = 42) {
  /* initialize */
  std::vector<T> h_a(h_matrix.cols), h_result(h_matrix.rows);
  /* random state */
  std::mt19937 gen(seed);
  /* run the experiment */
  for (int i = 0; i < matvec_product_cfg.num_experiments; i++) {
    if (i % 10 == 0) {
      printf(
          "Running backward error experiment : %d/%d for matrix of size : "
          "(%lu, %lu) \n",
          i + 1, matvec_product_cfg.num_experiments, h_matrix.rows,
          h_matrix.cols);
    }
    /* sample the vector */
    sample_random_vector(h_a, matvec_product_cfg.prec, matvec_product_cfg.dist,
                         gen);  // a vector
    /* run the matrix vector product(s) */
    launch_matvec_product_kernel<T>(h_matrix, h_a, h_result,
                                    matvec_product_cfg.prec);
    for (auto &r : h_result) {
      printf("%f\n", r);
    }
  }
}

void run_matrix_vector_product_backward_error_experiment(
    const matvec_product_config &matvec_product_cfg) {
  /* initialization */
  std::vector<backward_error_result> results(1);

  Matrix<double> test_matrix;
  test_matrix.rows = 50;
  test_matrix.cols = 50;
  test_matrix.data.resize(2500);
  std::mt19937 gen(42);
  sample_random_vector(test_matrix.data, Double, Ones, gen);

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Square Matrix-vector product backward error analysis config "
            << std::string(10, '-') << std::endl;
  print_matvec_product_config(matvec_product_cfg);
  std::cout << std::string(50, '=') << std::endl;

  /* assert statements */
  assert(matvec_product_cfg.prec == matvec_product_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  /* test run */
  run_matvec_product_backward_error_experiment_given_matrix<double>(
      test_matrix, matvec_product_cfg, results[0]);

  /* /1* run the experiment for fixed matrix size *1/ */
  /* switch (matvec_product_cfg.prec) { */
  /*   case Double: */
  /*     run_matvec_product_backward_error_experiment_given_matrix<double>( */
  /*         test_matrix, matvec_product_cfg, results[0]); */
  /*     break; */
  /*   case Single: */
  /*     run_matvec_product_backward_error_experiment_given_matrix<float>( */
  /*         test_matrix, matvec_product_cfg, results[0]); */
  /*     break; */
  /*   case Half: */
  /*     run_matvec_product_backward_error_experiment_given_matrix<half>( */
  /*         test_matrix, matvec_product_cfg, results[0]); */
  /*     break; */
  /*   default: */
  /*     throw std::invalid_argument("invalid precision"); */
  /* } */
}
