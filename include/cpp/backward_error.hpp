#ifndef BACKWARD_ERROR_HPP
#define BACKWARD_ERROR_HPP

#include <vector>
#include "gamma.hpp"

/* backward error results */
struct backward_error_result {
    int n; //  characteristic dimension
    double backward_error_min; // min backward error
    double backward_error_max; // max backward error
    double backward_error_mean; // mean backward error
    gamma_result backward_error_bound;       // bounds
    double nnz_to_size_ratio = 1.0; // number or non-zero to some characteristic size
};

/* dot product backward error */
void compute_sequential_dot_product_backward_error(double result, double result_true, double result_true_abs, double *backward_error);
gamma_result compute_sequential_dot_product_backward_error_bound(const int vector_size, const gamma_config &gamma_cfg, bool verbose=false);;

/* matrix-vector product backward error */
void compute_matvec_product_backward_error(const std::vector<double> &result, const std::vector<double> &result_true, const std::vector<double> &result_true_abs, double *backward_error);
gamma_result compute_matvec_product_backward_error_bound(
    const int rows,
    const int cols,
    const gamma_config &gamma_cfg,
    bool verbose = false
    );

/* boundary value problem backward error */
template <typename T>
void compute_ode_backward_error(
    const int num_intervals,
    std::vector<T> &h_sub_diag,
    std::vector<T> &h_main_diag,
    std::vector<T> &h_super_diag,
    std::vector<T> &h_rhs,
    std::vector<T> &h_state,
    double *backward_error,
    Precision prec
    );

/* boundary value problem backward error bounds */
gamma_result compute_ode_backward_error_bound(
    const int num_intervals,
    const gamma_config &gamma_cfg,
    bool verbose = false
    );


#endif
