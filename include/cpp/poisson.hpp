#ifndef POISSON_HPP
#define POISSON_HPP

#include "definition.hpp"
#include "gamma.hpp"
#include <iostream>

// configuration
struct poisson_config {
  // int X_res = 33; // Number of points in x-direction
  // int Y_res = 64; // Number of points in x-direction
  int X_res = 5; // Number of points in x-direction
  int Y_res = 6; // Number of points in x-direction
  double etol = 1e-6; // Error tolerance
  int max_iter = 5000; // Maximum number of iterations
  Precision prec = Single; // precision for the solve
  int num_experiments = 100; // number of experiments (number of times RHS is sampled)
  gamma_config gamma_cfg; // bounds config
  int block_jacobi_tile_size = 64; // block jacobi tile size
  int matvect_tile_size = 64; // tile size for the matrix vector product
};

// poisson equation solver
void run_poisson_equation_experiments(Precision prec);

#endif
