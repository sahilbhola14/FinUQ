#ifndef DOT_PRODUCT_HPP
#define DOT_PRODUCT_HPP

#include "definition.hpp"
#include "gamma.hpp"

/* configuration for dot productcs*/
struct dot_product_config {
  Precision prec = Single; // precision for sampling random vectors
  Distribution dist=Normal; // distribution for the random vectors
  int num_experiments = 100; // number of experiments
  gamma_config gamma_cfg; // bounds config
  int block_dim = 256; // block dimension for (block) dot product
};

/* /1* dot product experiment *1/ */
/* void run_dot_product_backward_error_experiment(const dot_product_config &dot_product_cfg, const int n_min=10, const int n_max=100000); */
/* void run_dot_product_forward_error_experiment(const int vector_size, const dot_product_config &dot_product_cfg); */

/* experiments */
void run_all_dot_product_experiments(Precision prec);

#endif
