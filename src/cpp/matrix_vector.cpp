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

/* matvec product filename */
std::string make_matvec_product_filename(const std::string prefix,
                                         const matvec_product_config &cfg) {
  std::ostringstream ss;
  ss << prefix << "_matvec_product_" << to_string(cfg.prec) << "_prec"
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
 * run matrix vector product backward error experiment for fixed size
 */
template <typename T>
void run_matvec_product_backward_error_experiment_given_matrix(
    const Matrix<T> &h_matrix, const matvec_product_config &matvec_product_cfg,
    backward_error_result &result, const int seed = 42) {
  /* initialize */
  const int rows = h_matrix.rows;
  const int cols = h_matrix.cols;
  std::vector<T> h_a(cols), h_result(rows);
  std::vector<double> h_a_true(cols), h_result_true(rows);
  std::vector<double> h_a_true_abs(cols), h_result_true_abs(rows);
  std::vector<double> backward_error(matvec_product_cfg.num_experiments);

  Matrix<double> h_matrix_true, h_matrix_true_abs;
  h_matrix_true.rows = rows;
  h_matrix_true.cols = cols;
  h_matrix_true.data.resize(rows * cols);
  h_matrix_true_abs.rows = rows;
  h_matrix_true_abs.cols = cols;
  h_matrix_true_abs.data.resize(rows * cols);

  gamma_result backward_error_bound;
  vector_stats backward_error_stats;

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
    convert_vector_to_double(h_a, h_a_true);
    convert_vector_to_double(h_matrix.data, h_matrix_true.data);
    convert_vector_to_absolute_double(h_a, h_a_true_abs);
    convert_vector_to_absolute_double(h_matrix.data, h_matrix_true_abs.data);

    /* run the matrix vector product(s) */
    launch_matvec_product_kernel<T>(h_matrix, h_a, h_result,
                                    matvec_product_cfg.prec);
    launch_matvec_product_kernel<T>(h_matrix_true, h_a_true, h_result_true,
                                    matvec_product_cfg.prec);
    launch_matvec_product_kernel<T>(h_matrix_true_abs, h_a_true_abs,
                                    h_result_true_abs, matvec_product_cfg.prec);

    /* compute the backward error */
    std::vector<double> h_result_converted(rows);
    convert_vector_to_double(h_result, h_result_converted);
    compute_matvec_product_backward_error(h_result_converted, h_result_true,
                                          h_result_true_abs,
                                          &backward_error[i]);
  }

  /* compute the backward error statistics */
  backward_error_stats = get_vector_stats(backward_error);

  /* compute the backward error bound */
  backward_error_bound = compute_matvec_product_backward_error_bound(
      rows, cols, matvec_product_cfg.gamma_cfg);

  /* update result */
  result.n = rows;  // same rows and cols
  result.backward_error_min = backward_error_stats.min;
  result.backward_error_max = backward_error_stats.max;
  result.backward_error_mean = backward_error_stats.mean;
  result.backward_error_bound = backward_error_bound;
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

  /* save */
  std::string filename =
      make_matvec_product_filename("backward_error_result", matvec_product_cfg);
  write_backward_error_results_csv(results, filename);
}

/* load the matrix market data */
std::vector<Matrix<double>> get_matrix_market_data(
    std::string filename = "square_matrices.bin") {
  std::vector<Matrix<double>> matrices = load_matrices_bin(filename);
  return matrices;
}
