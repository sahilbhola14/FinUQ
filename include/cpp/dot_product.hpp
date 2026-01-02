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
};

/* dot product experiment */
void run_dot_product_backward_error_experiment(const dot_product_config &dot_product_cfg);
void run_dot_product_forward_error_experiment(const int vector_size, const dot_product_config &dot_product_cfg);

#endif
