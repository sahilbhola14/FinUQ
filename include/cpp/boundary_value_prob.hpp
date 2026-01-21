#ifndef BOUNDARY_VALUE_PROB_HPP
#define BOUNDARY_VALUE_PROB_HPP

#include <vector>
#include "definition.hpp"
#include "gamma.hpp"

struct bvp_parameters{
  std::vector<double> theta_one;
  std::vector<double> theta_two;
};

struct bvp_config{
  Precision prec = Single; // precision for computing
  gamma_config gamma_cfg; // bounds config
};

/* compute analytical qoi */
void compute_analytical_qoi(const int num_samples=100000000, bool verbose =false, const int seed=42);

/* compute analytical solution of the linear system */
void compute_analytical_state(std::vector<double> &state, const double theta_one, const double theta_two, bool verbose=false);

/* run ode experiment */
void run_ode_backward_error_experiment(
    const bvp_config &bvp_cfg,
    const int num_samples=20, // number of parameter samples
    const int seed=42
    );

#endif
