#include "poisson.hpp"

#include <cassert>
#include <random>

#include "cholesky.cuh"
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
    // coeffient matrix
    std::vector<T> h_coeff = poisson.get_coefficient_matrix();
    // initial state
    std::vector<T> h_state_initial = poisson.get_initial_state();

    // jacobi solver
    launch_jacobi_solver<T>(poisson_rhs_cfg, h_coeff, h_state_initial,
                            poisson_cfg.prec, poisson_cfg.etol,
                            poisson_cfg.max_iter, true);
  }
}

// Cholesky per Tile
// Returns one Matrix<T> per block-diagonal tile, each storing the
// lower-triangular L
template <typename T>
std::vector<Matrix<T>> compute_cholesky_per_jacobi_tile(
    const std::vector<T> &h_a, const poisson_config &poisson_cfg) {
  const int state_dim = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
  const int tile_size = poisson_cfg.blk_jacobi_tile_size;
  const int num_tiles = state_dim / tile_size;

  std::vector<Matrix<T>> chol_factors(num_tiles);

  for (int t = 0; t < num_tiles; t++) {
    const int tile_start = t * tile_size;

    // (1) extract the tile_size x tile_size block-diagonal block
    std::vector<T> tile(tile_size * tile_size);
    for (int r = 0; r < tile_size; r++) {
      for (int c = 0; c < tile_size; c++) {
        tile[r * tile_size + c] =
            h_a[(tile_start + r) * state_dim + (tile_start + c)];
      }
    }

    // (2) compute Cholesky in prec_cholesky precision, result cast back to T
    const int tile_elems = tile_size * tile_size;
    std::vector<T> l(tile_elems, static_cast<T>(0.0));

    switch (poisson_cfg.prec_cholesky) {
      case Double: {
        std::vector<double> tile_d(tile_elems), l_d(tile_elems, 0.0);
        convert_vector_to_double(tile, tile_d);
        launch_cholesky_decomposition_kernel(tile_size, tile_d, l_d, Double);
        for (int k = 0; k < tile_elems; k++) l[k] = static_cast<T>(l_d[k]);
        break;
      }
      case Single: {
        std::vector<float> tile_f(tile_elems), l_f(tile_elems, 0.0f);
        convert_vector_to_float(tile, tile_f);
        launch_cholesky_decomposition_kernel(tile_size, tile_f, l_f, Single);
        for (int k = 0; k < tile_elems; k++) l[k] = static_cast<T>(l_f[k]);
        break;
      }
      case Half: {
        std::vector<half> tile_h(tile_elems),
            l_h(tile_elems, static_cast<half>(0.0f));
        convert_vector_to_half(tile, tile_h);
        launch_cholesky_decomposition_kernel(tile_size, tile_h, l_h, Half);
        for (int k = 0; k < tile_elems; k++) l[k] = static_cast<T>(l_h[k]);
        break;
      }
      default:
        throw std::invalid_argument("invalid prec_cholesky");
    }

    // (3) store in Matrix struct (same precision as h_a); nnz =
    // lower-triangular entries
    chol_factors[t].rows = tile_size;
    chol_factors[t].cols = tile_size;
    chol_factors[t].nnz = (tile_size * (tile_size + 1)) / 2;
    chol_factors[t].data = std::move(l);
  }

  return chol_factors;
}

// Block jacobi experiments (fixed discretization)
template <typename T>
void run_block_jacobi_experiments_fixed_discretization(
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
    // coeffient matrix
    std::vector<T> h_coeff = poisson.get_coefficient_matrix();
    // initial state
    std::vector<T> h_state_initial = poisson.get_initial_state();
    // cholesky decomp
    std::vector<Matrix<T>> chol_factors =
        compute_cholesky_per_jacobi_tile(h_coeff, poisson_cfg);

    // // print chol_factors
    // std::cout << std::scientific << std::setprecision(4);
    // for (int t = 0; t < static_cast<int>(chol_factors.size()); t++) {
    //   const Matrix<T> &L = chol_factors[t];
    //   std::cout << "Tile " << t << " L (" << L.rows << "x" << L.cols <<
    //   "):\n"; for (int r = 0; r < static_cast<int>(L.rows); r++) {
    //     for (int c = 0; c < static_cast<int>(L.cols); c++) {
    //       std::cout << std::setw(14) << static_cast<double>(L.data[r * L.cols
    //       + c]);
    //     }
    //     std::cout << "\n";
    //   }
    // }
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
  if (((poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2)) %
          poisson_cfg.blk_jacobi_tile_size !=
      0) {
    const int coeff_size = (poisson_cfg.X_res - 2) * (poisson_cfg.Y_res - 2);
    std::cout << "[DEBUG] Coeff size = " << coeff_size
              << ", Tile size = " << poisson_cfg.blk_jacobi_tile_size
              << ", Remainder = "
              << (coeff_size % poisson_cfg.blk_jacobi_tile_size) << std::endl;
    throw std::invalid_argument(
        "Coefficient matrix size: not divisible by tile size (Required for "
        "Block Jacobi)");
  }

  // print the header
  std::cout << std::string(50, '=') << std::endl;
  std::cout << std::string(10, '-')
            << " Block Jacobi solver for Poisson equation config "
            << std::string(10, '-') << std::endl;
  print_poisson_config(poisson_cfg);
  std::cout << "Block Jacobi Tile size: " << poisson_cfg.blk_jacobi_tile_size
            << std::endl;
  std::cout << "Block Jacobi Mat-vec Tile size: "
            << poisson_cfg.blk_jacobi_matvect_tile_size << std::endl;
  std::cout << std::string(50, '=') << std::endl;
  /* assert statements */
  assert(poisson_cfg.prec == poisson_cfg.gamma_cfg.prec &&
         "Bound precision and compute precision must be the same");

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
void run_all_block_jacobi_experiments(Precision prec,
                                      const int num_experiments = 100) {
  // configuration
  poisson_config poisson_cfg;

  poisson_cfg.prec = prec;
  poisson_cfg.prec_cholesky =
      prec;  // TODO: for mixed-precision, make it smaller

  poisson_cfg.num_experiments = num_experiments;
  poisson_cfg.gamma_cfg.prec = prec;        // bound precision
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
void run_poisson_equation_experiments(Precision prec) {
  run_all_jacobi_experiments(prec, 1);
  // run_all_block_jacobi_experiments(prec, 1);
}
