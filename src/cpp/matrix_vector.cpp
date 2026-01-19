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

  /* gamma_result backward_error_bound; */
  vector_stats backward_error_stats;
  gamma_result backward_error_bound;

  /* /1* random state *1/ */
  std::mt19937 gen(seed);

  /* run the experiment */
  for (int i = 0; i < matvec_product_cfg.num_experiments; i++) {
    if (i % 10 == 0) {
      printf("Running backward error experiment : %d/%d \n", i + 1,
             matvec_product_cfg.num_experiments);
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
    launch_matvec_product_kernel<double>(h_matrix_true, h_a_true, h_result_true,
                                         Double);
    launch_matvec_product_kernel<double>(h_matrix_true_abs, h_a_true_abs,
                                         h_result_true_abs, Double);

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
    const matvec_product_config &matvec_product_cfg,
    const std::string matrix_data_file) {
  /* load matrix-market data in double precision */
  std::vector<Matrix<double>> matrices =
      get_matrix_market_data(matrix_data_file);
  /* initialization */
  std::vector<backward_error_result> results(matrices.size());
  printf("%lu", matrices.size());

  /* print header */
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Square Matrix-vector product backward error analysis config "
            << std::string(10, '-') << std::endl;
  print_matvec_product_config(matvec_product_cfg);
  std::cout << "Number of marices from Matrix-market: " << matrices.size()
            << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* assert statements */
  assert(matvec_product_cfg.prec == matvec_product_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  /* run the experiments for each element of the matrix market matrix */
  for (int i = 0; i < matrices.size(); i++) {
    printf("Matrix : %d\n", i + 1);
    /* save the number of non-zero */
    results[i].nnz_to_size_ratio = static_cast<double>(matrices[i].nnz) /
                                   (static_cast<double>(matrices[i].rows) *
                                    static_cast<double>(matrices[i].cols));
    /* run */
    switch (matvec_product_cfg.prec) {
      case Double: {
        /* copy the sorce matrix and convert type */
        Matrix<double> matrix_double;
        copy_matrix_and_convert_precision(matrices[i], matrix_double);
        /* run experiment */
        run_matvec_product_backward_error_experiment_given_matrix<double>(
            matrix_double, matvec_product_cfg, results[i]);
        break;
      }
      case Single: {
        /* copy the sorce matrix and convert type */
        Matrix<float> matrix_float;
        copy_matrix_and_convert_precision(matrices[i], matrix_float);
        /* run experiment */
        run_matvec_product_backward_error_experiment_given_matrix<float>(
            matrix_float, matvec_product_cfg, results[i]);
        break;
      }
      case Half: {
        /* copy the sorce matrix and convert type */
        Matrix<half> matrix_half;
        copy_matrix_and_convert_precision(matrices[i], matrix_half);

        /* run experiment */
        run_matvec_product_backward_error_experiment_given_matrix<half>(
            matrix_half, matvec_product_cfg, results[i]);
        break;
      }
      default: {
        throw std::invalid_argument("invalid precision");
      }
    }
  }

  /* save */
  std::string filename =
      make_matvec_product_filename("backward_error_result", matvec_product_cfg);
  write_backward_error_results_csv(results, filename);
}

namespace matvec {
void run_all_backward_error_experiments(Precision prec,
                                        const int num_experiments = 100) {
  /* configuration */
  std::string matrix_data_file =
      "square_matrices_full.bin";  // matrix data file
  matvec_product_config matvec_product_cfg;
  matvec_product_cfg.prec = prec;            // sampling precision
  matvec_product_cfg.gamma_cfg.prec = prec;  // bound precision
  matvec_product_cfg.num_experiments =
      num_experiments;                             // number of experiments
  matvec_product_cfg.gamma_cfg.confidence = 0.99;  // overall confidence
  // beta shape parameter
  matvec_product_cfg.gamma_cfg.beta_dist_beta = 2.0;
  // alpha shape parameter
  std::vector<double> beta_dist_alpha_vals = {1.6, 1.7, 1.8,
                                              1.9};  // shape param. alpha

  /* data: U(0,1) */
  matvec_product_cfg.dist = ZeroOne;

  /* matvec_product_cfg.gamma_cfg.bound_model = Uniform; */
  /* run_matrix_vector_product_backward_error_experiment(matvec_product_cfg, */
  /*                                                     matrix_data_file); */

  matvec_product_cfg.gamma_cfg.bound_model = Beta;
  for (auto &alpha : beta_dist_alpha_vals) {
    matvec_product_cfg.gamma_cfg.beta_dist_alpha = alpha;
    run_matrix_vector_product_backward_error_experiment(matvec_product_cfg,
                                                        matrix_data_file);
  }

  /* data: U(-1,1) */
  matvec_product_cfg.dist = MinusOnePlusOne;

  /* matvec_product_cfg.gamma_cfg.bound_model = Uniform; */
  /* run_matrix_vector_product_backward_error_experiment(matvec_product_cfg, */
  /*                                                     matrix_data_file); */

  matvec_product_cfg.gamma_cfg.bound_model = Beta;
  for (auto &alpha : beta_dist_alpha_vals) {
    matvec_product_cfg.gamma_cfg.beta_dist_alpha = alpha;
    run_matrix_vector_product_backward_error_experiment(matvec_product_cfg,
                                                        matrix_data_file);
  }
}
}  // namespace matvec

/* run all experiments */
void run_all_matrix_vector_product_experiments(Precision prec) {
  /* run all backward error experiments */
  matvec::run_all_backward_error_experiments(prec);
}
