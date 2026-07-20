#include "forward_error.hpp"

#include <cuda_fp16.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "backward_error.hpp"
#include "block_jacobi.cuh"
#include "poisson.hpp"
#include "prob_model.hpp"
#include "rounding_error_model.cuh"
#include "utils.hpp"

namespace {

Matrix<double> eigen_to_matrix(const Eigen::MatrixXd &src) {
  Matrix<double> dst;
  dst.rows = static_cast<size_t>(src.rows());
  dst.cols = static_cast<size_t>(src.cols());
  dst.nnz = dst.rows * dst.cols;
  dst.data.resize(dst.nnz);
  for (int r = 0; r < src.rows(); r++) {
    for (int c = 0; c < src.cols(); c++) {
      dst.data[r * src.cols() + c] = src(r, c);
    }
  }
  return dst;
}

Eigen::MatrixXd matrix_to_eigen(const Matrix<double> &src) {
  Eigen::MatrixXd dst(src.rows, src.cols);
  for (size_t r = 0; r < src.rows; r++) {
    for (size_t c = 0; c < src.cols; c++) {
      dst(static_cast<int>(r), static_cast<int>(c)) =
          src.data[r * src.cols + c];
    }
  }
  return dst;
}

}  // namespace

template <typename T>
correction_matrix_result compute_block_jacobi_forcing_vector_impl(
    const std::vector<T> &h_coeff, const std::vector<T> &h_rhs,
    const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  const int tile_size = poisson_cfg.blk_jacobi_tile_size;
  if (state_dim <= 0 || tile_size <= 0) {
    return {};
  }
  if (state_dim % tile_size != 0) {
    throw std::invalid_argument(
        "state dimension must be divisible by blk_jacobi_tile_size");
  }
  if (static_cast<int>(h_coeff.size()) != state_dim * state_dim) {
    throw std::invalid_argument(
        "h_coeff size must equal the flattened dense coefficient matrix size");
  }
  if (static_cast<int>(h_rhs.size()) != state_dim) {
    throw std::invalid_argument("h_rhs size must equal the state dimension");
  }

  const std::vector<Matrix<T>> h_chol_factors =
      compute_cholesky_per_jacobi_tile(h_coeff, poisson_cfg);
  const correction_matrix_result g =
      compute_correction_G_matrix(h_coeff, h_chol_factors, poisson_cfg);
  const correction_matrix_result h =
      compute_correction_H_matrix(h_coeff, h_chol_factors, poisson_cfg);
  const block_jacobi_bound_coefficients_result coeffs =
      compute_block_jacobi_bound_coefficients(h_coeff, h_chol_factors,
                                              poisson_cfg);

  if (g.size() != 3 || h.size() != 3) {
    throw std::runtime_error(
        "block Jacobi correction helpers must return three bound-model "
        "matrices");
  }

  Eigen::MatrixXd a(state_dim, state_dim);
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd d_inv = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd d_inv_abs = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::VectorXd rhs_abs(state_dim);
  for (int r = 0; r < state_dim; r++) {
    rhs_abs(r) = std::abs(static_cast<double>(h_rhs[r]));
    for (int c = 0; c < state_dim; c++) {
      a(r, c) = static_cast<double>(h_coeff[r * state_dim + c]);
    }
  }

  for (int tile_start = 0; tile_start < state_dim; tile_start += tile_size) {
    Eigen::MatrixXd d_tile(tile_size, tile_size);
    for (int r = 0; r < tile_size; r++) {
      for (int c = 0; c < tile_size; c++) {
        d_tile(r, c) = static_cast<double>(
            h_coeff[(tile_start + r) * state_dim + (tile_start + c)]);
      }
    }
    d.block(tile_start, tile_start, tile_size, tile_size) = d_tile;
    const Eigen::MatrixXd d_tile_inv = d_tile.inverse();
    d_inv.block(tile_start, tile_start, tile_size, tile_size) = d_tile_inv;
    d_inv_abs.block(tile_start, tile_start, tile_size, tile_size) =
        d_tile_inv.cwiseAbs();
  }

  const Eigen::MatrixXd n = a - d;
  const Eigen::MatrixXd d_inv_n_abs = (d_inv * n).cwiseAbs();
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(state_dim, state_dim);
  const Eigen::MatrixXd i_minus_d_inv_n_abs = identity - d_inv_n_abs;
  const Eigen::VectorXd d_inv_abs_b = d_inv_abs * rhs_abs;

  const long double u_s =
      static_cast<long double>(compute_unit_roundoff(poisson_cfg.prec));
  const long double alpha_values[3] = {coeffs.alpha_s.gamma_det,
                                       coeffs.alpha_s.gamma_mprea,
                                       coeffs.alpha_s.gamma_vprea};
  const long double beta_values[3] = {coeffs.beta_s.gamma_det,
                                      coeffs.beta_s.gamma_mprea,
                                      coeffs.beta_s.gamma_vprea};
  const long double eta_values[3] = {coeffs.eta_s.gamma_det,
                                     coeffs.eta_s.gamma_mprea,
                                     coeffs.eta_s.gamma_vprea};

  correction_matrix_result forcing_vectors(3);
  for (int i = 0; i < 3; i++) {
    const Eigen::MatrixXd h_i = matrix_to_eigen(h[i]);
    const Eigen::MatrixXd g_i = matrix_to_eigen(g[i]);
    const Eigen::MatrixXd first_term =
        (static_cast<double>(u_s) * identity +
         static_cast<double>(beta_values[i] * alpha_values[i]) * h_i) *
        i_minus_d_inv_n_abs.inverse();
    const Eigen::MatrixXd second_term =
        static_cast<double>(eta_values[i]) * (identity - g_i).inverse();
    forcing_vectors[i] =
        eigen_to_matrix((first_term + second_term) * d_inv_abs_b);
  }

  return forcing_vectors;
}

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

/*
 * G_ii = (\gamma_{T;c}^2 + 3 \gamma_{T+1;c}) |A_ii^-1| |hat{R}_ii^T||hat{R}_ii|
 * gamma_factor = (\gamma_{T;c}^2 + 3 \gamma_{T+1;c})
 * G is blkdiag(G_ii)
 * for computing the probabilistic bounds, max_iteations is used to obtain the
 * probabilitity Q_solve = Q((k+1)(nT^2/6 + 3nT/2 + 4n/3 + n^2 + 1;\zeta).
 * condition to be satisfied: gamma_factor max_i (||A_ii^-1||_\infty
 * ||R_ii^T||_\infty||R_ii||_\infty) < 1 for all i.
 */
template <typename T>
correction_matrix_result compute_correction_G_matrix(
    const std::vector<T> &h_coeff, const std::vector<Matrix<T>> &h_chol_factors,
    const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  const int tile_size = poisson_cfg.blk_jacobi_tile_size;
  const int num_tiles = state_dim / tile_size;
  if (state_dim <= 0 || tile_size <= 0 || num_tiles <= 0) {
    return {};
  }
  if (static_cast<int>(h_chol_factors.size()) != num_tiles) {
    throw std::invalid_argument(
        "h_chol_factors size must match the number of block Jacobi tiles");
  }
  if (static_cast<int>(h_coeff.size()) != state_dim * state_dim) {
    throw std::invalid_argument(
        "h_coeff size must equal the flattened dense coefficient matrix size");
  }

  const long long state_dim_ll = static_cast<long long>(state_dim);
  const long long tile_size_ll = static_cast<long long>(tile_size);
  const long long per_iteration_bounds =
      (state_dim_ll * tile_size_ll * tile_size_ll) / 6LL +
      (3LL * state_dim_ll * tile_size_ll) / 2LL + (4LL * state_dim_ll) / 3LL +
      state_dim_ll * state_dim_ll + 1LL;
  const long long number_of_bounds =
      static_cast<long long>(poisson_cfg.max_iter + 1) * per_iteration_bounds;
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      static_cast<int>(number_of_bounds), poisson_cfg.gamma_cfg.confidence);

  gamma_config gamma_cfg_cholesky = poisson_cfg.gamma_cfg;
  gamma_cfg_cholesky.prec = poisson_cfg.prec_cholesky;
  gamma_result gamma_T =
      get_gamma(tile_size, gamma_cfg_cholesky, one_minus_zeta);
  gamma_result gamma_Tp =
      get_gamma(tile_size + 1, gamma_cfg_cholesky, one_minus_zeta);
  const gamma_result gamma_factor = gamma_T * gamma_T + 3.0L * gamma_Tp;

  Eigen::MatrixXd g_det = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd g_mprea = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd g_vprea = Eigen::MatrixXd::Zero(state_dim, state_dim);
  long double max_condition_det = 0.0L;
  long double max_condition_mprea = 0.0L;
  long double max_condition_vprea = 0.0L;

  for (int t = 0; t < num_tiles; t++) {
    Eigen::MatrixXd a_tile(tile_size, tile_size);
    Eigen::MatrixXd r_tile(tile_size, tile_size);
    const int tile_start = t * tile_size;

    for (int r = 0; r < tile_size; r++) {
      for (int c = 0; c < tile_size; c++) {
        a_tile(r, c) = static_cast<double>(
            h_coeff[(tile_start + r) * state_dim + (tile_start + c)]);
        r_tile(r, c) =
            static_cast<double>(h_chol_factors[t].data[r * tile_size + c]);
      }
    }

    const Eigen::MatrixXd a_inv_abs = a_tile.inverse().cwiseAbs();
    const double a_inv_inf_norm =
        a_inv_abs.cwiseAbs().rowwise().sum().maxCoeff();
    const double rt_inf_norm =
        r_tile.transpose().cwiseAbs().rowwise().sum().maxCoeff();
    const double r_inf_norm = r_tile.cwiseAbs().rowwise().sum().maxCoeff();
    const long double tile_condition =
        static_cast<long double>(a_inv_inf_norm) *
        static_cast<long double>(rt_inf_norm) *
        static_cast<long double>(r_inf_norm);
    max_condition_det =
        std::max(max_condition_det, gamma_factor.gamma_det * tile_condition);
    max_condition_mprea = std::max(max_condition_mprea,
                                   gamma_factor.gamma_mprea * tile_condition);
    max_condition_vprea = std::max(max_condition_vprea,
                                   gamma_factor.gamma_vprea * tile_condition);

    const Eigen::MatrixXd base_tile =
        a_inv_abs * (r_tile.transpose().cwiseAbs() * r_tile.cwiseAbs());
    g_det.block(tile_start, tile_start, tile_size, tile_size) =
        static_cast<double>(gamma_factor.gamma_det) * base_tile;
    g_mprea.block(tile_start, tile_start, tile_size, tile_size) =
        static_cast<double>(gamma_factor.gamma_mprea) * base_tile;
    g_vprea.block(tile_start, tile_start, tile_size, tile_size) =
        static_cast<double>(gamma_factor.gamma_vprea) * base_tile;
  }

  if (max_condition_det >= 1.0L || max_condition_mprea >= 1.0L ||
      max_condition_vprea >= 1.0L) {
    throw std::runtime_error(
        "Block Jacobi correction condition violated: "
        "gamma_factor * max_i(||A_ii^-1||_inf ||R_ii^T||_inf ||R_ii||_inf) "
        "must be < 1");
  }

  correction_matrix_result g(3);
  g[0] = eigen_to_matrix(g_det);
  g[1] = eigen_to_matrix(g_mprea);
  g[2] = eigen_to_matrix(g_vprea);
  return g;
}

/*
 * H = (I - G)^-1 |D^-1| |A|.
 * I: Identity matrix
 * G: obtained from compute_correction_G_matrix
 * D: blkdiag(D_ii), where D_ii is the diagonal block of the h_coeff of
 * tile_size*tile_size A: h_coeff matrix.
 */
template <typename T>
correction_matrix_result compute_correction_H_matrix(
    const std::vector<T> &h_coeff, const std::vector<Matrix<T>> &h_chol_factors,
    const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  const int tile_size = poisson_cfg.blk_jacobi_tile_size;
  const int num_tiles = state_dim / tile_size;
  if (state_dim <= 0 || tile_size <= 0 || num_tiles <= 0) {
    return {};
  }
  if (static_cast<int>(h_coeff.size()) != state_dim * state_dim) {
    throw std::invalid_argument(
        "h_coeff size must equal the flattened dense coefficient matrix size");
  }

  const correction_matrix_result g =
      compute_correction_G_matrix(h_coeff, h_chol_factors, poisson_cfg);
  if (g.size() != 3) {
    throw std::runtime_error(
        "compute_correction_G_matrix must return 3 matrices");
  }
  const Eigen::MatrixXd g_det = matrix_to_eigen(g[0]);
  const Eigen::MatrixXd g_mprea = matrix_to_eigen(g[1]);
  const Eigen::MatrixXd g_vprea = matrix_to_eigen(g[2]);

  Eigen::MatrixXd a_abs(state_dim, state_dim);
  for (int r = 0; r < state_dim; r++) {
    for (int c = 0; c < state_dim; c++) {
      a_abs(r, c) = std::abs(static_cast<double>(h_coeff[r * state_dim + c]));
    }
  }

  Eigen::MatrixXd d_inv_abs_a = Eigen::MatrixXd::Zero(state_dim, state_dim);
  for (int t = 0; t < num_tiles; t++) {
    const int tile_start = t * tile_size;
    Eigen::MatrixXd d_tile(tile_size, tile_size);
    for (int r = 0; r < tile_size; r++) {
      for (int c = 0; c < tile_size; c++) {
        d_tile(r, c) = static_cast<double>(
            h_coeff[(tile_start + r) * state_dim + (tile_start + c)]);
      }
    }
    const Eigen::MatrixXd d_inv_abs = d_tile.inverse().cwiseAbs();
    d_inv_abs_a.block(tile_start, 0, tile_size, state_dim) =
        d_inv_abs * a_abs.block(tile_start, 0, tile_size, state_dim);
  }

  const Eigen::MatrixXd i_minus_g_det =
      Eigen::MatrixXd::Identity(state_dim, state_dim) - g_det;
  const Eigen::MatrixXd i_minus_g_mprea =
      Eigen::MatrixXd::Identity(state_dim, state_dim) - g_mprea;
  const Eigen::MatrixXd i_minus_g_vprea =
      Eigen::MatrixXd::Identity(state_dim, state_dim) - g_vprea;

  correction_matrix_result h(3);
  h[0] = eigen_to_matrix(i_minus_g_det.inverse() * d_inv_abs_a);
  h[1] = eigen_to_matrix(i_minus_g_mprea.inverse() * d_inv_abs_a);
  h[2] = eigen_to_matrix(i_minus_g_vprea.inverse() * d_inv_abs_a);
  return h;
}

/*
 * \alpha_s: (1 + u_c)*(1 + u_s + \gamma_{m;s} + u_s * \gamma_{m;s} +
 * u_s*\gamma_m;s}) \beta_s : (1 + u_s) \eta_s: (1 + u_c) * (1 + u_s) * (1 +
 * u_s)
 * u_s: unit roundoff for prec in poisson_cfg u_c: unit roundoff for
 * prec_cholesky in poissoncfg
 *
 */
template <typename T>
block_jacobi_bound_coefficients_result compute_block_jacobi_bound_coefficients(
    const std::vector<T> &h_coeff, const std::vector<Matrix<T>> &h_chol_factors,
    const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  const int tile_size = poisson_cfg.blk_jacobi_tile_size;
  if (state_dim <= 0 || tile_size <= 0) {
    return {};
  }
  const int num_tiles = state_dim / tile_size;
  if (num_tiles <= 0) {
    return {};
  }
  if (static_cast<int>(h_chol_factors.size()) != num_tiles) {
    throw std::invalid_argument(
        "h_chol_factors size must match the number of block Jacobi tiles");
  }
  if (static_cast<int>(h_coeff.size()) != state_dim * state_dim) {
    throw std::invalid_argument(
        "h_coeff size must equal the flattened dense coefficient matrix size");
  }

  const long long state_dim_ll = static_cast<long long>(state_dim);
  const long long tile_size_ll = static_cast<long long>(tile_size);
  const long long per_iteration_bounds =
      (state_dim_ll * tile_size_ll * tile_size_ll) / 6LL +
      (3LL * state_dim_ll * tile_size_ll) / 2LL + (4LL * state_dim_ll) / 3LL +
      state_dim_ll * state_dim_ll + 1LL;
  const long long number_of_bounds =
      static_cast<long long>(poisson_cfg.max_iter + 1) * per_iteration_bounds;
  long double one_minus_zeta = compute_individual_bound_one_minus_zeta(
      static_cast<int>(number_of_bounds), poisson_cfg.gamma_cfg.confidence);

  const int matvec_tile_size = poisson_cfg.blk_jacobi_matvect_tile_size;
  if (matvec_tile_size <= 0) {
    throw std::invalid_argument(
        "blk_jacobi_matvect_tile_size must be positive");
  }

  const int m = matvec_tile_size + state_dim / matvec_tile_size;

  const gamma_result gamma_ms =
      get_gamma(m, poisson_cfg.gamma_cfg, one_minus_zeta);
  const long double u_s =
      static_cast<long double>(compute_unit_roundoff(poisson_cfg.prec));
  const long double u_c = static_cast<long double>(
      compute_unit_roundoff(poisson_cfg.prec_cholesky));

  const long double one_plus_u_s = 1.0L + u_s;
  const long double one_plus_u_c = 1.0L + u_c;

  block_jacobi_bound_coefficients_result result;

  // alpha coefficient
  result.alpha_s = {
      gamma_ms.n,
      one_plus_u_c * (one_plus_u_s + gamma_ms.gamma_det +
                      u_s * gamma_ms.gamma_det + u_s * gamma_ms.gamma_det),
      one_plus_u_c * (one_plus_u_s + gamma_ms.gamma_mprea +
                      u_s * gamma_ms.gamma_mprea + u_s * gamma_ms.gamma_mprea),
      one_plus_u_c * (one_plus_u_s + gamma_ms.gamma_vprea +
                      u_s * gamma_ms.gamma_vprea + u_s * gamma_ms.gamma_vprea)};

  // beta coefficient
  result.beta_s = {0, one_plus_u_s, one_plus_u_s, one_plus_u_s};

  // eta coefficient
  const long double eta_scalar = one_plus_u_c * one_plus_u_s * one_plus_u_s;
  result.eta_s = {0, eta_scalar, eta_scalar, eta_scalar};

  return result;
}

/*
 * P = \beta_s (I + \alpha_s H)
 * I: Identity matrix
 * H: obtained from compute_correction_H_matrix
 * \alpha_s: obtained from compute_block_jacobi_constants
 * \beta_s : obtained from compute_block_jacobi_constants
 * \eta_s: obtained from compute_block_jacobi_constants
 */
template <typename T>
correction_matrix_result compute_block_jacobi_P_matrix(
    const std::vector<T> &h_coeff, const std::vector<Matrix<T>> &h_chol_factors,
    const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  if (state_dim <= 0) {
    return {};
  }

  const correction_matrix_result h =
      compute_correction_H_matrix(h_coeff, h_chol_factors, poisson_cfg);
  if (h.size() != 3) {
    throw std::runtime_error(
        "compute_correction_H_matrix must return three correction matrices");
  }

  const block_jacobi_bound_coefficients_result coeffs =
      compute_block_jacobi_bound_coefficients(h_coeff, h_chol_factors,
                                              poisson_cfg);
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(state_dim, state_dim);

  correction_matrix_result p(3);
  const long double alpha_values[3] = {coeffs.alpha_s.gamma_det,
                                       coeffs.alpha_s.gamma_mprea,
                                       coeffs.alpha_s.gamma_vprea};
  const long double beta_values[3] = {coeffs.beta_s.gamma_det,
                                      coeffs.beta_s.gamma_mprea,
                                      coeffs.beta_s.gamma_vprea};

  for (int i = 0; i < 3; i++) {
    const Eigen::MatrixXd h_i = matrix_to_eigen(h[i]);
    const Eigen::MatrixXd p_i =
        static_cast<double>(beta_values[i]) *
        (identity + static_cast<double>(alpha_values[i]) * h_i);
    p[i] = eigen_to_matrix(p_i);
  }

  return p;
}

/*
 * forcing vector f: [(u_s I + \beta_s \alpha_s * H) (I - |D^{-1} N|)^{-1} +
 * \eta_s (I - G)^-1] |D^{-1}| |b| u_s: unit roundoff for prec in poisson_cfg
 * u_c: unit roundoff for prec_cholesky in poissoncfg I : Identity matrix
 * \alpha_s: obtained from compute_block_jacobi_constants
 * \beta_s: obtained from compute_block_jacobi_constants
 * \eta_s: obtained from compute_block_jacobi_constants
 * H: obtained from compute_correction_H_matrix
 * D: blkdiag(D_ii), where D_ii is the diagonal block of the h_coeff of
 * tile_size*tile_size.
 * A: h_coeff matrix.
 * A = D + N, where N is the matrix constructed using the off-diagonal blocks
 * G: obtained from compute_correction_G_matrix
 * b:
 */
template <typename T>
void compute_block_jacobi_forcing_vector(const std::vector<T> &h_coeff,
                                         const std::vector<T> &h_rhs,
                                         const poisson_config &poisson_cfg) {
  static_cast<void>(
      compute_block_jacobi_forcing_vector_impl(h_coeff, h_rhs, poisson_cfg));
}

// compute the asymptotic bounds
// bounds = (I - P)^{-1} f
// P: compute_block_jacobi_P_matrix
// f: compute_block_jacobi_forcing_vector
template <typename T>
correction_matrix_result compute_asymptotic_bounds(
    const std::vector<T> &h_coeff, const std::vector<T> &h_rhs,
    const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  if (state_dim <= 0) {
    return {};
  }

  const std::vector<Matrix<T>> h_chol_factors =
      compute_cholesky_per_jacobi_tile(h_coeff, poisson_cfg);
  const correction_matrix_result p =
      compute_block_jacobi_P_matrix(h_coeff, h_chol_factors, poisson_cfg);
  const correction_matrix_result f =
      compute_block_jacobi_forcing_vector_impl(h_coeff, h_rhs, poisson_cfg);

  if (p.size() != 3 || f.size() != 3) {
    throw std::runtime_error(
        "compute_asymptotic_bounds requires three bound-model matrices");
  }

  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(state_dim, state_dim);
  correction_matrix_result bounds(3);
  for (int i = 0; i < 3; i++) {
    const Eigen::MatrixXd p_i = matrix_to_eigen(p[i]);
    const double p_inf_norm = p_i.cwiseAbs().rowwise().sum().maxCoeff();

    assert(p_inf_norm < 1.0 &&
           "compute_asymptotic_bounds requires ||P||_inf < 1");
    const Eigen::MatrixXd f_i = matrix_to_eigen(f[i]);
    bounds[i] = eigen_to_matrix((identity - p_i).inverse() * f_i);
  }
  return bounds;
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

template correction_matrix_result compute_correction_G_matrix(
    const std::vector<double> &, const std::vector<Matrix<double>> &,
    const poisson_config &);
template correction_matrix_result compute_correction_G_matrix(
    const std::vector<float> &, const std::vector<Matrix<float>> &,
    const poisson_config &);
template correction_matrix_result compute_correction_G_matrix(
    const std::vector<half> &, const std::vector<Matrix<half>> &,
    const poisson_config &);

template correction_matrix_result compute_correction_H_matrix(
    const std::vector<double> &, const std::vector<Matrix<double>> &,
    const poisson_config &);
template correction_matrix_result compute_correction_H_matrix(
    const std::vector<float> &, const std::vector<Matrix<float>> &,
    const poisson_config &);
template correction_matrix_result compute_correction_H_matrix(
    const std::vector<half> &, const std::vector<Matrix<half>> &,
    const poisson_config &);

template block_jacobi_bound_coefficients_result
compute_block_jacobi_bound_coefficients(const std::vector<double> &,
                                        const std::vector<Matrix<double>> &,
                                        const poisson_config &);
template block_jacobi_bound_coefficients_result
compute_block_jacobi_bound_coefficients(const std::vector<float> &,
                                        const std::vector<Matrix<float>> &,
                                        const poisson_config &);
template block_jacobi_bound_coefficients_result
compute_block_jacobi_bound_coefficients(const std::vector<half> &,
                                        const std::vector<Matrix<half>> &,
                                        const poisson_config &);

template correction_matrix_result compute_block_jacobi_P_matrix(
    const std::vector<double> &, const std::vector<Matrix<double>> &,
    const poisson_config &);
template correction_matrix_result compute_block_jacobi_P_matrix(
    const std::vector<float> &, const std::vector<Matrix<float>> &,
    const poisson_config &);
template correction_matrix_result compute_block_jacobi_P_matrix(
    const std::vector<half> &, const std::vector<Matrix<half>> &,
    const poisson_config &);

template void compute_block_jacobi_forcing_vector(const std::vector<double> &,
                                                  const std::vector<double> &,
                                                  const poisson_config &);
template void compute_block_jacobi_forcing_vector(const std::vector<float> &,
                                                  const std::vector<float> &,
                                                  const poisson_config &);
template void compute_block_jacobi_forcing_vector(const std::vector<half> &,
                                                  const std::vector<half> &,
                                                  const poisson_config &);

template correction_matrix_result compute_asymptotic_bounds(
    const std::vector<double> &, const std::vector<double> &,
    const poisson_config &);
template correction_matrix_result compute_asymptotic_bounds(
    const std::vector<float> &, const std::vector<float> &,
    const poisson_config &);
template correction_matrix_result compute_asymptotic_bounds(
    const std::vector<half> &, const std::vector<half> &,
    const poisson_config &);
