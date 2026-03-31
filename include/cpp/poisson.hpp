#ifndef POISSON_HPP
#define POISSON_HPP

#include "definition.hpp"
#include "gamma.hpp"
#include "poisson_cuda.cuh"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// configuration
struct poisson_config {
  // int X_res = 33; // Number of points in x-direction
  // int Y_res = 64; // Number of points in y-direction
  int X_res = 5; // Number of points in x-direction
  int Y_res = 6; // Number of points in y-direction
  double etol = 1e-6; // Error tolerance
  int max_iter = 5000; // Maximum number of iterations
  Precision prec = Half; // precision for the solve
  int num_experiments = 100; // number of experiments (number of times RHS is sampled)
  gamma_config gamma_cfg; // bounds config
  int block_jacobi_tile_size = 64; // block jacobi tile size
  int matvect_tile_size = 64; // tile size for the matrix vector product
};

// Poisson class
template <typename T>
class Poisson {
 private:
  int get_state_dim() const { return Nx * Ny; }
  double get_x_resolution() const { return 1.0 / (Nx + 1); }
  double get_y_resolution() const { return 1.0 / (Ny + 1); }

  T get_inv_dx_sq() const {
    double dx = get_x_resolution();
    return static_cast<T>(1.0 / (dx * dx));
  }

  T get_inv_dy_sq() const {
    double dy = get_y_resolution();
    return static_cast<T>(1.0 / (dy * dy));
  }

  T get_diagonal_coefficient() const {
    T inv_dx_sq = get_inv_dx_sq();
    T inv_dy_sq = get_inv_dy_sq();
    return static_cast<T>(2.0) * (inv_dx_sq + inv_dy_sq);
  }

  T get_horizontal_offdiagonal_coefficient() const {
    T inv_dx_sq = get_inv_dx_sq();
    return static_cast<T>(-1.0) * inv_dx_sq;
  }
  T get_vertical_offdiagonal_coefficent() const {
    T inv_dy_sq = get_inv_dy_sq();
    return static_cast<T>(-1.0) * inv_dy_sq;
  }

  // rhs eval plus top boundary condition contribution
  std::vector<T> eval_rhs_host(const std::vector<T> &state) const {
    const int n = get_state_dim();
    std::vector<T> rhs(n);
    poisson_rhs_config<T> poisson_rhs_cfg = get_rhs_config();
    eval_rhs(poisson_rhs_cfg, state.data(), rhs.data());
    return rhs;
  }

 public:
  const int Nx;   // number of internal points in x direction
  const int Ny;   // number of internal points in y direction
  const T zeta;   // zeta value for the rhs

  Poisson(const poisson_config &cfg, const T zeta)
      : Nx(cfg.X_res - 2), Ny(cfg.Y_res - 2), zeta(zeta) {
    if (zeta <= static_cast<T>(0.0)) {
      throw std::invalid_argument("zeta must be greater than 0");
    }
  }

  int state_dim() const { return get_state_dim(); }

  std::vector<T> get_initial_state() const {
    return std::vector<T>(get_state_dim(), static_cast<T>(1.0));
  }

  poisson_rhs_config<T> get_rhs_config() const {
    return {zeta, get_inv_dy_sq(), Nx, get_state_dim()};
  }


  std::vector<T> get_coefficient_matrix() const {
    T diag_coeff              = get_diagonal_coefficient();
    T horizontal_offdiag_coeff = get_horizontal_offdiagonal_coefficient();
    T vertical_offdiag_coeff   = get_vertical_offdiagonal_coefficent();
    const int n = get_state_dim();
    std::vector<T> coeff(n * n, static_cast<T>(0.0));
    for (int i = 0; i < n; i++) {
      int d = i * n + i;
      coeff[d] = diag_coeff;
      if (i > 0)     coeff[d - 1] = horizontal_offdiag_coeff;
      if (i < n - 1) coeff[d + 1] = horizontal_offdiag_coeff;
      if (i > Nx - 1)    coeff[d - Nx] = vertical_offdiag_coeff;
      if (i < n - Nx)    coeff[d + Nx] = vertical_offdiag_coeff;
    }
    // enforce left boundary (zero out wrap-around off-diagonals)
    for (int i = Nx; i < n; i += Nx) {
      coeff[i * n + i - 1] = static_cast<T>(0.0);
    }
    // enforce right boundary
    for (int i = Nx - 1; i < n; i += Nx) {
      coeff[i * n + i + 1] = static_cast<T>(0.0);
    }
    return coeff;
  }

  void print_coefficient_matrix() const {
    const int n = get_state_dim();
    std::vector<T> coeff = get_coefficient_matrix();
    std::cout << std::scientific << std::setprecision(4);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        std::cout << std::setw(14) << static_cast<double>(coeff[i * n + j]);
      }
      std::cout << std::endl;
    }
  }

  void print_rhs(const std::vector<T> &state) const {
    for (auto &a : eval_rhs_host(state)) {
      std::cout << static_cast<double>(a) << std::endl;
    }
  }
};

// Poisson equation solver
void run_poisson_equation_experiments(Precision prec);

#endif
