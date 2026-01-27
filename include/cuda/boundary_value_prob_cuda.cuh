#ifndef BOUNDARY_VALUE_PROB_CUDA_CUH
#define BOUNDARY_VALUE_PROB_CUDA_CUH

#include <vector>

#include "definition.hpp"
#include "gamma.hpp"

/* |\hat{L}||\hat{U}| kernel */
template <typename T>
void launch_abs_lu_multiplication_kernel(
    const int num_intervals, const std::vector<T> &h_sub_diag,
    const std::vector<T> &h_main_diag, const std::vector<T> &h_super_diag,
    const std::vector<T> &h_rhs, std::vector<double> &h_abs_lu_mult_true,
    Precision prec, bool verbose = false);

/* thomas algorithm kernel */
template <typename T>
void launch_thomas_algorithm_kernel(const int num_intervals,
                                    const std::vector<T> &h_sub_diag,
                                    const std::vector<T> &h_main_diag,
                                    const std::vector<T> &h_super_diag,
                                    const std::vector<T> &h_rhs,
                                    std::vector<T> &h_state, Precision prec,
                                    bool verbose = false);

/* thomas algorithm model kernel */
void launch_thomas_algorithm_model_kernel(
    const int num_intervals, const std::vector<double> &h_sub_diag,
    const std::vector<double> &h_main_diag,
    const std::vector<double> &h_super_diag, const std::vector<double> &h_rhs,
    std::vector<double> &h_state, Precision prec, const gamma_config &gamma_cfg,
    const int experiment_id, bool verbose = false);

/* ode state integral kernel */
template <typename T>
void launch_ode_state_integral_kernel(const int num_intervals,
                                      const std::vector<T> &h_sub_diag,
                                      const std::vector<T> &h_main_diag,
                                      const std::vector<T> &h_super_diag,
                                      const std::vector<T> &h_rhs,
                                      T &h_state_integral, Precision prec,
                                      bool verbose = false);

/* ode state integral model kernel */
void launch_ode_state_integral_model_kernel(
    const int num_intervals, const std::vector<double> &h_sub_diag,
    const std::vector<double> &h_main_diag,
    const std::vector<double> &h_super_diag, const std::vector<double> &h_rhs,
    double &h_state_integral, Precision prec, const gamma_config &gamma_cfg,
    const int experiment_id, bool verbose = false);

/* integrate the state (obtained from Thomas) kernel */
template <typename T>
void launch_state_integral_kernel(const int num_intervals,
                                  std::vector<T> &h_state, T &h_state_integral,
                                  Precision prec, bool verbose = false);

void launch_state_integral_model_kernel(
    const int num_intervals, std::vector<double> &h_state,
    double &h_state_integral, Precision prec, const gamma_config &gamma_cfg,
    const int experiment_id, bool verbose = false);

/* monte carlo expectation kernel */
template <typename T>
void launch_monte_carlo_expectation_kernel(const std::vector<T> &h_integrand,
                                           T &h_integral, Precision prec,
                                           bool verbose = false);

/* monte carlo expectation model kernel */
void launch_monte_carlo_expectation_model_kernel(
    const std::vector<double> &h_integrand, double &h_integral, Precision prec,
    const gamma_config &gamma_cfg, const int experiment_id,
    bool verbose = false);

/* kernel tags */
enum class BVPKernelTag : int {
  LUDecomposition,
  ForwardSubstitution,
  BackwardSubstitution,
  StateIntegral,
  MonteCarlo,
  Count
};

#endif
