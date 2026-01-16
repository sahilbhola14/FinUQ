#include "boundary_value_prob.hpp"

#include <random>

#include "distribution.hpp"

bvp_parameters sample_bvp_parameters(const int num_samples, std::mt19937 &gen) {
  /* initialize */
  bvp_parameters bvp_params;
  bvp_params.theta_one.resize(num_samples);
  bvp_params.theta_two.resize(num_samples);
  /* sample theta_1 ~ U(0.1, 1.1) */
  sample_uniform_distribution(bvp_params.theta_one, 0.1, 1.1, gen);
  /* /1* sample theta_2 ~ U(1, 2) *1/ */
  sample_uniform_distribution(bvp_params.theta_two, 1.0, 2.0, gen);
  return bvp_params;
}

/* compute analytical state integral */
void compute_analytical_state_integral(double &integral, const double theta_one,
                                       const double theta_two) {
  double numerator =
      25.0 * std::pow(theta_two, 2.0) *
      (-2.0 * theta_one + (2.0 + theta_one) * std::log1p(theta_one));
  double denominator = std::pow(theta_one, 2.0) * std::log1p(theta_one);
  integral = numerator / denominator;
}

/* compute analytical qoi */
void compute_analytical_qoi(const int num_mcmc_samples, bool verbose,
                            const int seed) {
  /* initialize */
  double state_integral, mean;
  /* random generator */
  std::mt19937 gen(42);
  /* sample the parameters */
  bvp_parameters bvp_params = sample_bvp_parameters(num_mcmc_samples, gen);
  /* compute the state integral */
  for (int i = 0; i < num_mcmc_samples; i++) {
    compute_analytical_state_integral(state_integral, bvp_params.theta_one[i],
                                      bvp_params.theta_two[i]);
    if (i == 0) {
      mean = state_integral;
    } else {
      mean = mean + (state_integral - mean) / i;
    }
  }
  /* print */
  if (verbose == true)
    printf("Analytical QoI (q): %.3e (using %d MCMC samples)\n", mean,
           num_mcmc_samples);
}
