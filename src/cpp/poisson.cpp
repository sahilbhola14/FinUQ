#include "poisson.hpp"

#include <cassert>
#include <random>

#include "block_jacobi.cuh"
#include "cholesky.cuh"
#include "distribution.hpp"
#include "prob_model.hpp"
#include "utils.hpp"

namespace {

int get_poisson_state_dim(const poisson_config &poisson_cfg) {
  return (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
}

void assert_valid_block_jacobi_tiling(const poisson_config &poisson_cfg) {
  const int state_dim = get_poisson_state_dim(poisson_cfg);
  assert(poisson_cfg.blk_jacobi_tile_size > 0 &&
         "blk_jacobi_tile_size must be positive");
  assert(poisson_cfg.blk_jacobi_matvect_tile_size > 0 &&
         "blk_jacobi_matvect_tile_size must be positive");
  assert(state_dim >= poisson_cfg.blk_jacobi_tile_size &&
         "state dimension must be >= blk_jacobi_tile_size");
  assert(state_dim >= poisson_cfg.blk_jacobi_matvect_tile_size &&
         "state dimension must be >= blk_jacobi_matvect_tile_size");
  assert(state_dim % poisson_cfg.blk_jacobi_tile_size == 0 &&
         "state dimension must be divisible by blk_jacobi_tile_size");
  assert(state_dim % poisson_cfg.blk_jacobi_matvect_tile_size == 0 &&
         "state dimension must be divisible by blk_jacobi_matvect_tile_size");
}

}  // namespace

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
// for each experiment, a new \zeta value is sampled
// the solver solves \nabla^2 u - \zeta u = f.
template <typename T>
void run_jacobi_experiments_fixed_discretization(
    const poisson_config &poisson_cfg) {
  // sample the zeta values
  std::vector<T> zeta_vals(poisson_cfg.num_experiments);
  std::mt19937 gen(/*seed=*/42);
  // sample_uniform_distribution(zeta_vals, 10.0, 10.0, gen);
  sample_uniform_distribution(zeta_vals, 0.0, 0.0, gen);

  // run the experiment
  for (int i = 0; i < poisson_cfg.num_experiments; i++) {
    // initialize the poisson object
    Poisson<T> poisson(poisson_cfg, zeta_vals[i]);
    // rhs
    std::vector<T> h_rhs = poisson.get_rhs();
    // coeffient matrix
    std::vector<T> h_coeff = poisson.get_coefficient_matrix();
    // initial state
    std::vector<T> h_state_initial = poisson.get_initial_state();
    // solver config
    poisson_solver_config<T> poisson_solver_cfg = poisson.get_solver_config();
    // poisson.print_coefficient_matrix();
    // jacobi solver
    // launch_jacobi_solver<T>(h_coeff, h_state_initial, h_rhs,
    // poisson_cfg.prec,
    //                         poisson_cfg.etol, poisson_cfg.max_iter,
    //                         poisson_solver_cfg, true);
    //
    // compute cholesky factors for each block
    std::vector<Matrix<T>> h_chol_factors =
        compute_cholesky_per_jacobi_tile(h_coeff, poisson_cfg);

    // test the bound matrices
    const correction_matrix_result test = compute_correction_G_matrix(
        h_coeff, h_chol_factors, poisson_cfg, poisson_cfg.max_iter);

    // const correction_matrix_result asymptotic_bounds =
    //     compute_asymptotic_bounds(h_coeff, h_rhs, poisson_cfg);
    // const char *bound_labels[3] = {"Asymptotic bound (deterministic)",
    //                                "Asymptotic bound (mean-informed)",
    //                                "Asymptotic bound (variance-informed)"};
    // for (int j = 0; j < static_cast<int>(asymptotic_bounds.size()) && j < 3;
    //      j++) {
    //   print_matrix(asymptotic_bounds[j], bound_labels[j]);
    // }
  }
}

// Block jacobi experiments (fixed discretization)
template <typename T>
void run_block_jacobi_experiments_fixed_discretization(
    const poisson_config &poisson_cfg) {
  // sample the zeta values
  std::vector<T> zeta_vals(poisson_cfg.num_experiments);
  std::mt19937 gen(/*seed=*/42);
  sample_uniform_distribution(zeta_vals, 0.0, 0.0, gen);

  // run the experiment
  for (int i = 0; i < poisson_cfg.num_experiments; i++) {
    // initialize the poisson object
    Poisson<T> poisson(poisson_cfg, zeta_vals[i]);
    // rhs
    std::vector<T> h_rhs = poisson.get_rhs();
    // coefficient matrix
    std::vector<T> h_coeff = poisson.get_coefficient_matrix();
    // initial state
    std::vector<T> h_state_initial = poisson.get_initial_state();
    // solver config
    poisson_solver_config<T> poisson_solver_cfg = poisson.get_solver_config();
    // cholesky decomp
    std::vector<Matrix<T>> h_chol_factors =
        compute_cholesky_per_jacobi_tile(h_coeff, poisson_cfg);

    // // compute per iteration bounds
    // const correction_matrix_result per_iteration_bounds =
    //     compute_per_iteration_bounds(h_coeff, h_rhs, h_state_initial,
    //                                  poisson_cfg, poisson_cfg.max_iter);
    // print_matrix(per_iteration_bounds[0], "drea");
    // print_matrix(per_iteration_bounds[1], "mprea");
    // print_matrix(per_iteration_bounds[2], "vprea");

    // sweep per iteration bounds over all iterations and save to csv
    save_per_iteration_bounds(h_coeff, h_rhs, h_state_initial, poisson_cfg);

    // jacobi solver
    launch_block_jacobi_solver(h_coeff, h_state_initial, h_rhs, poisson_cfg,
                               poisson_solver_cfg, true);

    // // run the jacobi solver(s)
    // launch_jacobi_solver<T>(poisson_rhs_cfg, h_coeff, h_state_initial,
    //                         poisson_cfg.prec, poisson_cfg.etol,
    //                         poisson_cfg.max_iter, true);
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
  assert_valid_block_jacobi_tiling(poisson_cfg);

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

// block jacobi experiments
void run_block_jacobi_experiments(const poisson_config &poisson_cfg) {
  // initialization
  if (poisson_cfg.X_res <= 2 || poisson_cfg.Y_res <= 2) {
    throw std::invalid_argument("X_res and Y_res must be greater that 2");
  }

  // Cholesky must be performed at the same or lower precision than the solver
  // (lower precision = larger urd, i.e. higher enum value)
  assert(poisson_cfg.prec_cholesky >= poisson_cfg.prec &&
         "Cholesky precision must have urd >= solver precision urd "
         "(prec_cholesky must be <= precision of prec)");

  // print the header
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Block Jacobi solver for Poisson equation config "
            << std::string(10, '-') << std::endl;
  print_poisson_config(poisson_cfg);
  std::cout << "Cholesky precision: " << to_string(poisson_cfg.prec_cholesky)
            << std::endl;
  std::cout << "Block Jacobi Tile size: " << poisson_cfg.blk_jacobi_tile_size
            << std::endl;
  std::cout << "Block Jacobi Mat-vec Tile size: "
            << poisson_cfg.blk_jacobi_matvect_tile_size << std::endl;
  std::cout << std::string(50, '=') << std::endl;
  /* assert statements */
  assert(poisson_cfg.prec == poisson_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");
  assert_valid_block_jacobi_tiling(poisson_cfg);

  // run the experiment
  switch (poisson_cfg.prec) {
    case Double: {
      run_block_jacobi_experiments_fixed_discretization<double>(poisson_cfg);
      break;
    }
    case Single: {
      run_block_jacobi_experiments_fixed_discretization<float>(poisson_cfg);
      break;
    }
    case Half: {
      run_block_jacobi_experiments_fixed_discretization<half>(poisson_cfg);
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
  poisson_cfg.gamma_cfg.prec = prec;        // bound precision
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

// Block jacobi experiments all experiments
void run_all_block_jacobi_experiments(Precision prec, Precision prec_cholesky,
                                      const int num_experiments = 100) {
  // configuration
  poisson_config poisson_cfg;

  poisson_cfg.prec = prec;
  poisson_cfg.prec_cholesky = prec_cholesky;

  poisson_cfg.num_experiments =
      num_experiments;                // number of times the rhs is sampled
  poisson_cfg.gamma_cfg.prec = prec;  // bound precision
  poisson_cfg.gamma_cfg.confidence = 0.99;  // overall confidence
  // std::vector <int> block_jacobi_tile_sizes = {4, 8, 16, 32};
  std::vector<int> block_jacobi_tile_sizes = {4};

  // beta shape parameter
  poisson_cfg.gamma_cfg.beta_dist_beta = 2.0;
  // alpha shape parameter
  std::vector<double> beta_dist_alpha_vals = {1.8, 1.9,
                                              2.0};  // shape param. alpha

  // run the experiment (Uniform rounding error model)
  for (auto &tile_size : block_jacobi_tile_sizes) {
    poisson_cfg.gamma_cfg.bound_model = Uniform;
    poisson_cfg.blk_jacobi_tile_size = tile_size;

    run_block_jacobi_experiments(poisson_cfg);
  }

  // for (auto &alpha: beta_dist_alpha_vals){
  //   poisson_cfg.gamma_cfg.beta_dist_alpha = alpha;
  //   run_jacobi_experiments(poisson_cfg);
  // }
}

// poisson equation experiments
void run_poisson_equation_experiments(Precision prec, Precision prec_cholesky) {
  // run_all_jacobi_experiments(prec, 1);
  run_all_block_jacobi_experiments(prec, prec_cholesky, 1);
}
