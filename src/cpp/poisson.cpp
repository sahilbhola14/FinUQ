#include "poisson.hpp"

#include <cassert>
#include <random>

#include "distribution.hpp"
#include "prob_model.hpp"
#include "utils.hpp"

/* print poisson config */
void print_poisson_config(const poisson_config &poisson_cfg) {
  std::cout << "Compute precision: " << to_string(poisson_cfg.prec)
            << std::endl;
  std::cout << "X-resolution: " << poisson_cfg.X_res << std::endl;
  std::cout << "Y-resolution: " << poisson_cfg.Y_res << std::endl;
  std::cout << "Max iterations: " << poisson_cfg.max_iter << std::endl;
  std::cout << "Error tolerance: " << poisson_cfg.etol << std::endl;
  std::cout << "Number of experiments: " << poisson_cfg.num_experiments
            << std::endl;
  std::cout << "Bound model: " << to_string(poisson_cfg.gamma_cfg.bound_model)
            << std::endl;
  if (poisson_cfg.gamma_cfg.bound_model == Beta) {
    std::cout << "Beta bound model alpha value: "
              << poisson_cfg.gamma_cfg.beta_dist_alpha << std::endl;
    std::cout << "Beta bound model beta value: "
              << poisson_cfg.gamma_cfg.beta_dist_beta << std::endl;
    check_mean_rounding_error_sign(poisson_cfg.prec,
                                   poisson_cfg.gamma_cfg.bound_model,
                                   poisson_cfg.gamma_cfg.beta_dist_alpha,
                                   poisson_cfg.gamma_cfg.beta_dist_beta);
  }
  std::cout << "Bound confidence: " << std::setprecision(4)
            << poisson_cfg.gamma_cfg.confidence << std::endl;
}

// jacobi experiments (fixed discretization)
template <typename T>
void run_jacobi_experiments_fixed_discretization(
    const poisson_config &poisson_cfg) {
  // sample the zeta values
  std::vector<T> zeta_vals(poisson_cfg.num_experiments);
  std::mt19937 gen(/*seed=*/42);
  sample_uniform_distribution(zeta_vals, 0.0, 1.0, gen);

  // run the experiment
  for (int i = 0; i < poisson_cfg.num_experiments; i++) {
    // initialize the poisson object
    Poisson<T> poisson(poisson_cfg, zeta_vals[i]);
    poisson_rhs_config<T> poisson_rhs_cfg = poisson.get_rhs_config();
    std::vector<T> h_coeff = poisson.get_coefficient_matrix();
    std::vector<T> h_state_initial = poisson.get_initial_state();
    // run the jacobi solver(s)
    launch_jacobi_solver<T>(poisson_rhs_cfg, h_coeff, h_state_initial,
                            poisson_cfg.prec, poisson_cfg.etol,
                            poisson_cfg.max_iter, true);
  }
}

// jacobi experiments
void run_jacobi_experiments(const poisson_config &poisson_cfg) {
  // initialization
  if (poisson_cfg.X_res <= 2 || poisson_cfg.Y_res <= 2) {
    throw std::invalid_argument("X_res and Y_res must be greater that 2");
  }
  // print the header
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Jacobi solver for Poisson equation config "
            << std::string(10, '-') << std::endl;
  print_poisson_config(poisson_cfg);
  std::cout << std::string(50, '=') << std::endl;
  /* assert statements */
  assert(poisson_cfg.prec == poisson_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

  // run the experiment
  switch (poisson_cfg.prec) {
    case Double: {
      run_jacobi_experiments_fixed_discretization<double>(poisson_cfg);
      break;
    }
    case Single: {
      run_jacobi_experiments_fixed_discretization<float>(poisson_cfg);
      break;
    }
    case Half: {
      run_jacobi_experiments_fixed_discretization<half>(poisson_cfg);
      break;
    }
    default: {
      throw std::invalid_argument("invalid precision");
    }
  }
}

// jacobi experiments all experiments
void run_all_jacobi_experiments(Precision prec,
                                const int num_experiments = 100) {
  // configuration
  poisson_config poisson_cfg;
  poisson_cfg.prec = prec;
  poisson_cfg.num_experiments = num_experiments;
  poisson_cfg.gamma_cfg.confidence = 0.99;  // overall confidence
  // beta shape parameter
  poisson_cfg.gamma_cfg.beta_dist_beta = 2.0;
  // alpha shape parameter
  std::vector<double> beta_dist_alpha_vals = {1.8, 1.9,
                                              2.0};  // shape param. alpha
  // run the experiment (Uniform rounding error model)
  poisson_cfg.gamma_cfg.bound_model = Uniform;
  run_jacobi_experiments(poisson_cfg);

  // for (auto &alpha: beta_dist_alpha_vals){
  //   poisson_cfg.gamma_cfg.beta_dist_alpha = alpha;
  //   run_jacobi_experiments(poisson_cfg);
  // }
}

// poisson equation experiments
void run_poisson_equation_experiments(Precision prec) {
  run_all_jacobi_experiments(prec, 1);
}
