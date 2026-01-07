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
