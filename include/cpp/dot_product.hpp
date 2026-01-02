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

/* dot product results */
struct dot_product_result {
    int n; //  vector size
    double mean_backward_error; // mean backward error
    double mean_forward_error; // mean forward error
    double max_backward_error; // max backward error
    double max_forward_error; // max forward error
    double bound_backward_error; // bound for the backward error
    double bound_forward_error; // bound for the forward error
};

/* dot product experiment */
void run_dot_product_experiment(const dot_product_config &dot_product_cfg);

#endif
