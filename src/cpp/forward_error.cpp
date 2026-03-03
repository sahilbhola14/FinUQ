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

/*compute the forward error bonds for the boundary value problem state
 * compute |u - \hat{u}| \leq \Gamma |A^{-1}| |A| |\hat{u}| where the bounds
 * are satisfied with probability Q(M + 8 * Ns - 6). The probaiblity assumes
 * that the state perturbation is propagated to compute the QoI later. M :
 * Number of monte carlo samples Ns : Number of intervals - 1, that is, the
 * state size.
 */
template <typename T>
std::vector<gamma_result> compute_bvp_state_forward_error_bound(
    const int num_intervals, const int num_samples,
    const std::vector<T> &h_sub_diag, const std::vector<T> &h_main_diag,
    const std::vector<T> &h_super_diag, const std::vector<T> &h_state,
    const gamma_config &gamma_cfg, bool verbose = false) {
  // initialization
  const int Ns = num_intervals - 1;  // state size
  const int M = num_samples;         // number of monte carlo samples
  const int number_of_bounds = M * (7 * Ns * Ns - 5 * Ns + 1);
  // compute individual bound one_minus_zeta
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      number_of_bounds, gamma_cfg.confidence);

  // compute gamma(s)
  gamma_result gamma_one = get_gamma(1, gamma_cfg, one_minus_zeta);
  gamma_result gamma_two = get_gamma(2, gamma_cfg, one_minus_zeta);
  gamma_result gamma_thomas =
      2.0 * gamma_one + gamma_two + gamma_one * gamma_two;

  // compute |A^{-1}||A||\hat{u}| in double precision
  std::vector<double> abs_a_inv_abs_a_abs_sol = compute_abs_a_inv_abs_a_abs_sol(
      num_intervals, h_sub_diag, h_main_diag, h_super_diag, h_state);

  // compute the bound
  std::vector<gamma_result> state_bounds;
  state_bounds.reserve(Ns);
  for (int i = 0; i < Ns; i++) {
    state_bounds[i] = abs_a_inv_abs_a_abs_sol[i] * gamma_thomas;
  }

  // print
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " State bounds for Number of intervals: " << num_intervals
              << " and Number of Monte-Carlo samples: " << num_samples << " "
              << std::string(10, '-') << std::endl;

    for (int i = 0; i < Ns; i++) {
      std::cout << "i = " << i << std::endl;
      print_gamma(state_bounds[i], true);
    }
  }

  return state_bounds;
}

/* compute the forward error bounds for the boundary value problem state
 * integral
 * compute |p - \hat{p}| \leq \Delta x \sum_{i=1}^{Ns} |\Delta u_i|,
 * wher \Delta u_i is the propagated perturbation.
 * */
template <typename T>
gamma_result compute_bvp_state_integral_forward_error_bound(
    const int num_intervals, const int num_samples,
    const std::vector<T> &h_sub_diag, const std::vector<T> &h_main_diag,
    const std::vector<T> &h_super_diag, const std::vector<T> &h_state,
    const gamma_config &gamma_cfg, bool verbose) {
  // initialization
  const int Ns = num_intervals - 1;  // state size
  const int M = num_samples;         // number of monte carlo samples
  // number of bounds to be satisfied
  const int number_of_bounds = M * (7 * Ns * Ns - 5 * Ns + 1);
  const double delta_x = 1.0 / num_intervals;  // discretization

  // compute individual bound one_minus_zeta
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      number_of_bounds, gamma_cfg.confidence);

  // compute gamma(s)
  gamma_result gamma_Ns = get_gamma(Ns, gamma_cfg, one_minus_zeta);

  // compute the bounds for the state aboluste forward error
  std::vector<gamma_result> state_bounds =
      compute_bvp_state_forward_error_bound(
          num_intervals, num_samples, h_sub_diag, h_main_diag, h_super_diag,
          h_state, gamma_cfg, verbose);

  // compute absolute state
  std::vector<double> state_abs;
  state_abs.reserve(Ns);
  for (int i = 0; i < Ns; i++) {
    state_abs.push_back(std::abs(static_cast<double>(h_state[i])));
  }

  // compute the bounds for the realization p, that is, the state integral
  gamma_result state_integral_bounds;
  for (int i = 0; i < Ns; i++) {
    state_integral_bounds =
        state_integral_bounds + (state_abs[i] + state_bounds[i]);
  }
  state_integral_bounds = delta_x * gamma_Ns * state_integral_bounds;

  // print
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " State integral bounds for Number of intervals: "
              << num_intervals
              << " and Number of Monte-Carlo samples: " << num_samples << " "
              << std::string(10, '-') << std::endl;
    print_gamma(state_integral_bounds, true);
  }

  return state_integral_bounds;
}

/* compute the forward error in qoi computation for the boundary value problem
 */
void compute_bvp_qoi_forward_error(double result, double result_true,
                                   double *forward_error, bool verbose) {
  *forward_error = std::abs(result - result_true);
  if (verbose == true) {
    printf("Absolute forward error in the Qoi: %.5e\n", *forward_error);
  }
}

/* compute the forward error bounds for the boundary value problem qoi
 */
template <typename T>
gamma_result compute_bvp_qoi_forward_error_bound(
    const int num_intervals, const int num_samples,
    const std::vector<T> &h_state_integral,
    const std::vector<gamma_result> &forward_error_bound_state_integral,
    const gamma_config &gamma_cfg, bool verbose) {
  // initialization
  const int Ns = num_intervals - 1;  // state size
  const int M = num_samples;         // number of monte carlo samples
  // number of bounds to be satisfied
  const int number_of_bounds = M * (7 * Ns * Ns - 5 * Ns + 1);
  const double delta_x = 1.0 / num_intervals;  // discretization

  // compute individual bound one_minus_zeta
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      number_of_bounds, gamma_cfg.confidence);

  // compute gamma
  gamma_result gamma_M = get_gamma(M, gamma_cfg, one_minus_zeta);

  // compute absolute state_integral
  std::vector<double> state_integral_abs;
  state_integral_abs.reserve(M);
  for (int i = 0; i < M; i++) {
    state_integral_abs.push_back(
        std::abs(static_cast<double>(h_state_integral[i])));
  }

  // compute the bounds for the qoi, that is, expected state integral
  gamma_result qoi_bounds;
  for (int i = 0; i < M; i++) {
    // print_gamma(forward_error_bound_state_integral[i]);
    qoi_bounds = qoi_bounds + (state_integral_abs[i] +
                               forward_error_bound_state_integral[i]);
  }
  qoi_bounds = (1.0 / M) * gamma_M * qoi_bounds;

  // print
  if (verbose == true) {
    std::cout << std::string(10, '-')
              << " QoI bounds for Number of intervals: " << num_intervals
              << " and Number of Monte-Carlo samples: " << num_samples << " "
              << std::string(10, '-') << std::endl;
    print_gamma(qoi_bounds, true);
  }

  return qoi_bounds;
}

/* template initialization */
template gamma_result compute_bvp_state_integral_forward_error_bound<double>(
    const int, const int, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    const std::vector<double> &, const gamma_config &, bool verbose);
template gamma_result compute_bvp_state_integral_forward_error_bound<float>(
    const int, const int, const std::vector<float> &,
    const std::vector<float> &, const std::vector<float> &,
    const std::vector<float> &, const gamma_config &, bool verbose);
template gamma_result compute_bvp_state_integral_forward_error_bound<half>(
    const int, const int, const std::vector<half> &, const std::vector<half> &,
    const std::vector<half> &, const std::vector<half> &, const gamma_config &,
    bool verbose);

template gamma_result compute_bvp_qoi_forward_error_bound<double>(
    const int, const int, const std::vector<double> &,
    const std::vector<gamma_result> &, const gamma_config &, bool);
template gamma_result compute_bvp_qoi_forward_error_bound<float>(
    const int, const int, const std::vector<float> &,
    const std::vector<gamma_result> &, const gamma_config &, bool);
template gamma_result compute_bvp_qoi_forward_error_bound<half>(
    const int, const int, const std::vector<half> &,
    const std::vector<gamma_result> &, const gamma_config &, bool);
