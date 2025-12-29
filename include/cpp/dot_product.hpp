#ifndef DOT_PRODUCT_HPP
#define DOT_PRODUCT_HPP

#include "definition.hpp"

/* configuration for dot productcs*/
/* struct dot_product_config { */
/*     int num_runs = 1000; // number of runs for a fixed vector size */
/*     Distribution dist=Normal; */
/* }; */

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
void run_dot_product_experiment(Precision prec, Distribution dist, const int num_experiments=1);

#endif
