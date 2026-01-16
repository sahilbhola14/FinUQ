#ifndef BOUNDARY_VALUE_PROB_HPP
#define BOUNDARY_VALUE_PROB_HPP

#include <vector>

struct bvp_parameters{
  std::vector<double> theta_one;
  std::vector<double> theta_two;
};

struct bvp_config{
  int Nx; // number of discretization points
  bvp_parameters bvp_params; // model parameters
};

/* compute analytical qoi */
void compute_analytical_qoi(const int num_mcmc_samples=100000000, bool verbose =false, const int seed=42);

/* compute analytical solution of the linear system */

#endif
