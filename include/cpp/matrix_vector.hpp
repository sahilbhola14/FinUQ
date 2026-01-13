#ifndef MATRIX_VECTOR_HPP
#define MATRIX_VECTOR_HPP

#include "definition.hpp"
#include "gamma.hpp"

/* configuration for dot productcs*/
struct matrix_vector_config {
  Precision prec = Single; // precision for sampling random vectors
  Distribution dist=Normal; // distribution for the random vectors
  int num_experiments = 100; // number of experiments
  gamma_config gamma_cfg; // bounds config
};

/* /1* dot product experiment *1/ */
/* void run_dot_product_backward_error_experiment(const dot_product_config &dot_product_cfg, const int n_min=10, const int n_max=100000); */
/* void run_dot_product_forward_error_experiment(const int vector_size, const dot_product_config &dot_product_cfg); */

/* /1* experiments *1/ */
/* void run_all_dot_product_experiments(Precision prec); */

void load_matrix_market_data();

#endif
