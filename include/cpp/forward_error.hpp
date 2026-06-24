#ifndef FORWARD_ERROR_HPP
#define FORWARD_ERROR_HPP

#include <vector>
#include "gamma.hpp"

template <typename T>
struct Matrix;

struct poisson_config;

/* forward error results */
struct forward_error_result {
    int n; //  characteristic dimension
    std::vector<double> forward_error; // forward error
    std::vector<double> forward_error_model; // forward error model
    std::vector<gamma_result> forward_error_bound; // forward error
};

using correction_matrix_result = std::vector<Matrix<double>>;

/* dot product forward error */
void compute_sequential_dot_product_forward_error(double result, double result_true, double *forward_error);
gamma_result compute_sequential_dot_product_forward_error_bound(const int vector_size, double result_true, double result_true_abs, const gamma_config &gamma_cfg, bool verbose=false);;

/* bvp forward error bound for the state integral */
template <typename T>
gamma_result compute_bvp_state_integral_forward_error_bound(
    const int num_intervals,
    const int num_samples,
    const std::vector<T> &h_sub_diag,
    const std::vector<T> &h_main_diag,
    const std::vector<T> &h_super_diag,
    const std::vector<T> &h_state,
    const gamma_config &gamma_cfg,
    bool verbose=false
    );

// bvp forward error for the qoi
void compute_bvp_qoi_forward_error(
    double result,
    double result_true,
    double *forward_error,
    bool verbose = false
    );

// bvp forward error bound for the qoi
template <typename T>
gamma_result compute_bvp_qoi_forward_error_bound(
    const int num_intervals,
    const int num_samples,
    const std::vector<T> &h_state_integral,
    const std::vector<gamma_result> &forward_error_bound_state_integral,
    const gamma_config &gamma_cfg,
    bool verbose=false
    );

// compute correction G matrix
template <typename T>
correction_matrix_result compute_correction_G_matrix(
    const std::vector<T> &h_coeff,
    const std::vector<Matrix<T>> &h_chol_factors,
    const poisson_config &poisson_cfg);

template <typename T>
correction_matrix_result compute_correction_H_matrix(
    const std::vector<T> &h_coeff,
    const std::vector<Matrix<T>> &h_chol_factors,
    const poisson_config &poisson_cfg);

#endif
