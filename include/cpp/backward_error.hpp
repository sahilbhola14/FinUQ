#ifndef BACKWARD_ERROR_HPP
#define BACKWARD_ERROR_HPP

#include "gamma.hpp"

/* backward error results */
struct backward_error_result {
    int n; //  characteristic dimension
    double backward_error_min; // mean backward error
    double backward_error_max; // max backward error
    double backward_error_mean; // mean backward error
    gamma_result gamma;       // bounds
};

/* dot product backward error */
void compute_sequential_dot_product_backward_error(double result, double result_true, double result_true_abs, double *backward_error);
gamma_result compute_sequential_dot_product_backward_error_bound(const int vector_size, const gamma_config &gamma_cfg, bool verbose=false);;
#endif
