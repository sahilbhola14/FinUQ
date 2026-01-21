/*
 * References
 * [1] Higham, N.J. and Mary, T., 2019. A new approach to probabilislic
 * rounding error analysis. SIAM journal on scientific computing, 41(5),
 * pp.A2815-A2835. [2]
 */
#include "backward_error.hpp"

#include <cmath>
#include <iostream>

#include "prob_model.hpp"
#include "utils.hpp"

/*
 * compute the dot-product backward error.
 *
 * @param result           Result of the computation: <a, b>
 * @param result_true      True result computed in double precision: <a_true,
 * b_true>
 * @param result_true_abs  True result computed in double precision using
 * absolute values: <|a_true|, |b_true|>
 */
void compute_sequential_dot_product_backward_error(double result,
                                                   double result_true,
                                                   double result_true_abs,
                                                   double *backward_error) {
  /* equation 4.1 in [1] */
  *backward_error = std::abs(result - result_true) / result_true_abs;
}

/*
 * compute the dot-product backward error bound.
 *
 * @param gamma_cfg   Configuration of the bounds
 */
gamma_result compute_sequential_dot_product_backward_error_bound(
    const int vector_size, const gamma_config &gamma_cfg, bool verbose) {
  /* compute individual bound confidence when number_of_bounds to be satisfied
   * is vector_size*/
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      vector_size, gamma_cfg.confidence);
  /* compute the bounds \gamma_{vector_size}*/
  gamma_result result = get_gamma(vector_size, gamma_cfg, one_minus_zeta);
  /* verbose */
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " Dot product backward error bounds for vector size : "
              << vector_size << " " << std::string(10, '-') << std::endl;
    std::cout << "Deterministic: " << result.gamma_det << std::endl;
    std::cout << "Mean-informed: " << result.gamma_mprea << std::endl;
    std::cout << "Varinance-informed: " << result.gamma_vprea << std::endl;
  }
  return result;
}

/*
 * compute the mat-vec product backward error
 * equation (4.3) in """ A NEW APPROACH TO PROBABILISTIC ROUNDING ERROR
 * ANALYSIS"""
 */
void compute_matvec_product_backward_error(
    const std::vector<double> &result, const std::vector<double> &result_true,
    const std::vector<double> &result_true_abs, double *backward_error) {
  /* initialize */
  double max = 0.0;
  double ratio = 0.0;

  /* compute the backward error */
  for (int i = 0; i < result.size(); i++) {
    ratio = std::abs(result[i] - result_true[i]) / result_true_abs[i];
    max = std::max(max, ratio);
  }

  /* save */
  *backward_error = max;
}

/* compute the mat-vec product backward error bound */
gamma_result compute_matvec_product_backward_error_bound(
    const int rows, const int cols, const gamma_config &gamma_cfg,
    bool verbose) {
  /* compute individual bound confidence when number_of_bounds to be satisfied
   * is rows* cols */
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      rows * cols, gamma_cfg.confidence);

  /* compute the bounds \gamma_{cols}*/
  gamma_result result = get_gamma(cols, gamma_cfg, one_minus_zeta);
  /* verbose */
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " Mat-vec product backward error bounds for matrix of size : ("
              << rows << ", " << cols << ") " << std::string(10, '-')
              << std::endl;
    std::cout << "Deterministic: " << result.gamma_det << std::endl;
    std::cout << "Mean-informed: " << result.gamma_mprea << std::endl;
    std::cout << "Varinance-informed: " << result.gamma_vprea << std::endl;
  }
  return result;
}

/* compute the boundary value problem backward error */
template <typename T>
void compute_ode_backward_error(const int num_intervals,
                                std::vector<T> &h_sub_diag,
                                std::vector<T> &h_main_diag,
                                std::vector<T> &h_super_diag,
                                std::vector<T> &h_rhs, std::vector<T> &h_state,
                                double *backward_error, Precision prec) {
  /* initialization */
  const int Ns = num_intervals - 1;
  std::vector<double> h_a_mat_times_state_true_minus_rhs(
      Ns);  // A \times state - rhs in double precision
  std::vector<double> h_a_mat_abs_times_abs_state_true(Ns);  //|A| * |state|

  /* compute A \times state - RHS and |A|\times|state|*/
  for (int i = 0; i < Ns; i++) {
    /* compute A \times state */
    if (i == 0) {
      h_a_mat_times_state_true_minus_rhs[i] =
          static_cast<double>(h_main_diag[i]) *
              static_cast<double>(h_state[i]) +
          static_cast<double>(h_super_diag[i]) *
              static_cast<double>(h_state[i + 1]);

      h_a_mat_abs_times_abs_state_true[i] =
          std::abs(static_cast<double>(h_main_diag[i])) *
              std::abs(static_cast<double>(h_state[i])) +
          std::abs(static_cast<double>(h_super_diag[i])) *
              std::abs(static_cast<double>(h_state[i + 1]));

    } else if (i == Ns - 1) {
      h_a_mat_times_state_true_minus_rhs[i] =
          static_cast<double>(h_sub_diag[i]) *
              static_cast<double>(h_state[i - 1]) +
          static_cast<double>(h_main_diag[i]) * static_cast<double>(h_state[i]);

      h_a_mat_abs_times_abs_state_true[i] =
          std::abs(static_cast<double>(h_sub_diag[i])) *
              std::abs(static_cast<double>(h_state[i - 1])) +
          std::abs(static_cast<double>(h_main_diag[i])) *
              std::abs(static_cast<double>(h_state[i]));

    } else {
      h_a_mat_times_state_true_minus_rhs[i] =
          static_cast<double>(h_sub_diag[i]) *
              static_cast<double>(h_state[i - 1]) +
          static_cast<double>(h_main_diag[i]) *
              static_cast<double>(h_state[i]) +
          static_cast<double>(h_super_diag[i]) *
              static_cast<double>(h_state[i + 1]);

      h_a_mat_abs_times_abs_state_true[i] =
          std::abs(static_cast<double>(h_sub_diag[i])) *
              std::abs(static_cast<double>(h_state[i - 1])) +
          std::abs(static_cast<double>(h_main_diag[i])) *
              std::abs(static_cast<double>(h_state[i])) +
          std::abs(static_cast<double>(h_super_diag[i])) *
              std::abs(static_cast<double>(h_state[i + 1]));
    }
    /* subtract RHS */
    h_a_mat_times_state_true_minus_rhs[i] =
        h_a_mat_times_state_true_minus_rhs[i] - static_cast<double>(h_rhs[i]);
  }

  /* compute backward error */
  double max = 0.0;
  double ratio = 0.0;
  for (int i = 0; i < Ns; i++) {
    ratio = std::abs(h_a_mat_times_state_true_minus_rhs[i]) /
            h_a_mat_abs_times_abs_state_true[i];
    max = std::max(max, ratio);
  }

  /* save */
  *backward_error = max;
}

/* compute the boundary value problem backward error bound*/
gamma_result compute_ode_backward_error_bound(const int num_intervals,
                                              const gamma_config &gamma_cfg,
                                              bool verbose) {
  const int Ns = num_intervals - 1;
  /* compute individual bound confidence when number_of_bounds to be satisfied
   * is 7*Ns - 6 */
  long double one_minus_zeta =
      compute_individual_bound_one_minus_zeta(7 * Ns - 6, gamma_cfg.confidence);
  /* compute the bounds 2.0\gamma_1 + \gamma_2 + \gamma_1\gamma_2*/
  gamma_result gamma1 = get_gamma(1, gamma_cfg, one_minus_zeta);
  gamma_result gamma2 = get_gamma(2, gamma_cfg, one_minus_zeta);

  gamma_result result;
  result.gamma_det = 2.0 * gamma1.gamma_det + gamma2.gamma_det +
                     gamma1.gamma_det * gamma2.gamma_det;
  result.gamma_mprea = 2.0 * gamma1.gamma_mprea + gamma2.gamma_mprea +
                       gamma1.gamma_mprea * gamma2.gamma_mprea;
  result.gamma_vprea = 2.0 * gamma1.gamma_vprea + gamma2.gamma_vprea +
                       gamma1.gamma_vprea * gamma2.gamma_vprea;

  /* verbose */
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " ODE backward error bounds for number of intervals : "
              << num_intervals << " " << std::string(10, '-') << std::endl;
    std::cout << "Deterministic: " << result.gamma_det << std::endl;
    std::cout << "Mean-informed: " << result.gamma_mprea << std::endl;
    std::cout << "Varinance-informed: " << result.gamma_vprea << std::endl;
  }
  return result;
}

/* template initialization */
template void compute_ode_backward_error<double>(
    const int num_intervals, std::vector<double> &h_sub_diag,
    std::vector<double> &h_main_diag, std::vector<double> &h_super_diag,
    std::vector<double> &h_rhs, std::vector<double> &h_state,
    double *backward_error, Precision);
template void compute_ode_backward_error<float>(
    const int num_intervals, std::vector<float> &h_sub_diag,
    std::vector<float> &h_main_diag, std::vector<float> &h_super_diag,
    std::vector<float> &h_rhs, std::vector<float> &h_state,
    double *backward_error, Precision);
template void compute_ode_backward_error<half>(
    const int num_intervals, std::vector<half> &h_sub_diag,
    std::vector<half> &h_main_diag, std::vector<half> &h_super_diag,
    std::vector<half> &h_rhs, std::vector<half> &h_state,
    double *backward_error, Precision);
