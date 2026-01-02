#ifndef FORWARD_ERROR_HPP
#define FORWARD_ERROR_HPP

#include <vector>
#include "gamma.hpp"

/* forward error results */
struct forward_error_result {
    int n; //  characteristic dimension
    std::vector<double> forward_error; // forward error
    std::vector<double> forward_error_model; // forward error model
    std::vector<gamma_result> forward_error_bound; // forward error
};

/* dot product forward error */
void compute_sequential_dot_product_forward_error(double result, double result_true, double *forward_error);
gamma_result compute_sequential_dot_product_forward_error_bound(const int vector_size, double result_true, double result_true_abs, const gamma_config &gamma_cfg, bool verbose=false);;
#endif
