#include "forward_error.hpp"

#include <cuda_fp16.h>

#include <Eigen/Dense>
#include <iostream>

#include "backward_error.hpp"
#include "prob_model.hpp"
/*
 * compute the dot-product forward error.
 *
 * @param result           Result of the computation: <a, b>
 * @param result_true_abs  True result computed in double precision using
 * absolute values: <|a_true|, |b_true|>
 */
void compute_sequential_dot_product_forward_error(double result,
                                                  double result_true,
                                                  double *forward_error) {
  *forward_error = std::abs(result - result_true) / std::abs(result_true);
}

/*
 * compute the dot-product forward error bound.
 *
 * @param vector_size   vector size
 * @param gamma_cfg   Configuration of the bounds
 * @param result_true      True result of the computation: <a, b>
 * @param result_true_abs  True result computed in double precision using
 * absolute values: <|a_true|, |b_true|>
 * @param gamma_cfg: configuration of the bounds
 */
gamma_result compute_sequential_dot_product_forward_error_bound(
    const int vector_size, double result_true, double result_true_abs,
    const gamma_config &gamma_cfg, bool verbose) {
  gamma_result backward_error, forward_error;
  double condition;
  /* compute the backward error bound */
  backward_error = compute_sequential_dot_product_backward_error_bound(
      vector_size, gamma_cfg);
  /* compute the condition */
  condition = result_true_abs / std::abs(result_true);
  /* compute the forward error bound */
  forward_error.n = vector_size;
  forward_error.gamma_det = backward_error.gamma_det * condition;
  forward_error.gamma_mprea = backward_error.gamma_mprea * condition;
  forward_error.gamma_vprea = backward_error.gamma_vprea * condition;

  /* verbose */
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " Dot product forward error bounds for vector size : "
              << vector_size << " " << std::string(10, '-') << std::endl;
    std::cout << "Deterministic: " << forward_error.gamma_det << std::endl;
    std::cout << "Mean-informed: " << forward_error.gamma_mprea << std::endl;
    std::cout << "Varinance-informed: " << forward_error.gamma_vprea
              << std::endl;
  }
  return forward_error;
}

/* compute |A^{-1}||A||\hat{u}| in double precision */
template <typename T>
std::vector<double> compute_abs_a_inv_abs_a_abs_sol(
    const int num_intervals, const std::vector<T> &h_sub_diag,
    const std::vector<T> &h_main_diag, const std::vector<T> &h_super_diag,
    const std::vector<T> &h_state) {
  const int Ns = num_intervals - 1;
  if (Ns <= 0) {
    return {};
  }

  std::vector<double> sub_diag(Ns), main_diag(Ns), super_diag(Ns);
  std::vector<double> state_abs(Ns), rhs(Ns);
  std::vector<double> abs_a_inv_abs_a_abs_sol(Ns);

  for (int i = 0; i < Ns; i++) {
    sub_diag[i] = static_cast<double>(h_sub_diag[i]);
    main_diag[i] = static_cast<double>(h_main_diag[i]);
    super_diag[i] = static_cast<double>(h_super_diag[i]);
    state_abs[i] = std::abs(static_cast<double>(h_state[i]));
  }

  // rhs = |A| * |state|
  for (int i = 0; i < Ns; i++) {
    const double a_abs = std::abs(sub_diag[i]);
    const double b_abs = std::abs(main_diag[i]);
    const double c_abs = std::abs(super_diag[i]);
    if (i == 0) {
      rhs[i] = b_abs * state_abs[i];
      if (Ns > 1) {
        rhs[i] += c_abs * state_abs[i + 1];
      }
    } else if (i == Ns - 1) {
      rhs[i] = a_abs * state_abs[i - 1] + b_abs * state_abs[i];
    } else {
      rhs[i] = a_abs * state_abs[i - 1] + b_abs * state_abs[i] +
               c_abs * state_abs[i + 1];
    }
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Ns, Ns);
  for (int i = 0; i < Ns; i++) {
    A(i, i) = main_diag[i];
    if (i > 0) {
      A(i, i - 1) = sub_diag[i];
    }
    if (i < Ns - 1) {
      A(i, i + 1) = super_diag[i];
    }
  }

  Eigen::MatrixXd A_inv_abs = A.inverse().cwiseAbs();
  Eigen::VectorXd rhs_vec(Ns);
  for (int i = 0; i < Ns; i++) {
    rhs_vec(i) = rhs[i];
  }
  Eigen::VectorXd result = A_inv_abs * rhs_vec;

  for (int i = 0; i < Ns; i++) {
    abs_a_inv_abs_a_abs_sol[i] = result(i);
  }
  return abs_a_inv_abs_a_abs_sol;
}

/* compute the forward error bounds for the boundary value problem state
 * integral */
template <typename T>
gamma_result compute_bvp_state_integral_forward_error_bound(
    const int num_intervals, const int num_samples,
    const std::vector<T> &h_sub_diag, const std::vector<T> &h_main_diag,
    const std::vector<T> &h_super_diag, const std::vector<T> &h_state,
    const T &h_state_integral, const gamma_config &gamma_cfg, bool verbose) {
  /* compute |A^{-1}||A||\hat{u}| in double precision */
  std::vector<double> abs_a_inv_abs_a_abs_sol = compute_abs_a_inv_abs_a_abs_sol(
      num_intervals, h_sub_diag, h_main_diag, h_super_diag, h_state);
  /* compute individual bound confidence when number_of_bounds to be satisfied
   * is (Ns + 8(M-1) - 6) */
  const int Ns = num_intervals - 1;
  const int number_of_bounds = num_samples + 8 * Ns - 6;
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      number_of_bounds, gamma_cfg.confidence);

  /* compute \gamma_{M-1} (|\hat{u}} + (2\gamma_1 + \gamma_2 + \gamma_1\gamma2)
   * ... abs_a_inv_abs_a_abs_sol), that is the bound for |Delta state| */
  gamma_result gamma1 = get_gamma(1, gamma_cfg, one_minus_zeta);
  gamma_result gamma2 = get_gamma(2, gamma_cfg, one_minus_zeta);
  gamma_result gammaNs =
      get_gamma(num_intervals - 1, gamma_cfg, one_minus_zeta);
  gamma_result gamma_thomas = 2.0 * gamma1 + gamma2 + gamma1 * gamma2;
  std::vector<gamma_result> DeltaU;
  DeltaU.reserve(Ns);
  double state_abs;
  for (int i = 0; i < Ns; i++) {
    state_abs = std::abs(static_cast<double>(h_state[i]));
    DeltaU.push_back((state_abs + abs_a_inv_abs_a_abs_sol[i] * gamma_thomas) *
                     gammaNs);
  }

  /* sum_{i=1}^Ns \DeltaU_i*/
  gamma_result sum_absDeltaU;
  for (int i = 0; i < Ns; i++) {
    sum_absDeltaU = sum_absDeltaU + gamma_abs(DeltaU[i]);
  }

  /* compute gamma_{num_samples} * (|state_integral| + \Delta x \sum_{i=1}^{Ns}
   * |Delta state|) that is the bounds for Delta p
   */
  const double delta_x = static_cast<double>(1.0 / num_intervals);
  gamma_result Deltap;
  gamma_result gamma_samples =
      get_gamma(num_samples, gamma_cfg, one_minus_zeta);
  Deltap = (std::abs(static_cast<double>(h_state_integral)) +
            delta_x * sum_absDeltaU) *
           gamma_samples;

  /* print */
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " Dot product forward error bounds for number of interval: "
              << num_intervals << " and number of samples: " << num_samples
              << " " << std::string(10, '-') << std::endl;
    std::cout << "Deterministic: " << Deltap.gamma_det << std::endl;
    std::cout << "Mean-informed: " << Deltap.gamma_mprea << std::endl;
    std::cout << "Varinance-informed: " << Deltap.gamma_vprea << std::endl;
  }

  return Deltap;
}

/* compute the forward error in qoi computation for the boundary value problem
 */
void compute_bvp_qoi_forward_error(double result, double result_true,
                                   double *forward_error) {
  *forward_error = std::abs(result - result_true);
}

/* template initialization */
template gamma_result compute_bvp_state_integral_forward_error_bound<double>(
    const int, const int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    const std::vector<double> &, const double &, const gamma_config &,
    bool verbose);
template gamma_result compute_bvp_state_integral_forward_error_bound<float>(
    const int, const int, const std::vector<float> &,
    const std::vector<float> &, const std::vector<float> &,
    const std::vector<float> &, const float &, const gamma_config &,
    bool verbose);
template gamma_result compute_bvp_state_integral_forward_error_bound<half>(
    const int, const int, const std::vector<half> &, const std::vector<half> &,
    const std::vector<half> &, const std::vector<half> &, const half &,
    const gamma_config &, bool verbose);
