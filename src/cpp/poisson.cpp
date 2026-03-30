#include "poisson.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Poisson class
template <typename T>
class Poisson {
 private:
  // state dim
  int get_state_dim() const { return Nx * Ny; }

  // x resolution: dx
  double get_x_resolution() const { return 1.0 / (Nx + 1); }

  // y resolution: dy
  double get_y_resolution() const { return 1.0 / (Ny + 1); }

  //  1 / (dx * dx)
  T get_inv_dx_sq() const {
    double dx = get_x_resolution();
    return static_cast<T>(1.0 / (dx * dx));
  }

  //  1 / (dy * dy)
  T get_inv_dy_sq() const {
    double dy = get_y_resolution();
    return static_cast<T>(1.0 / (dy * dy));
  }

  // diagonal elements: -2 * (inv_dx_sq + inv_dy_sq)
  T get_diagonal_coefficient() const {
    T inv_dx_sq = get_inv_dx_sq();
    T inv_dy_sq = get_inv_dy_sq();
    return -2.0 * (inv_dx_sq + inv_dy_sq);
  }

  // horizontal adjacent offdiagonals: inv_dx_sq
  T get_horizontal_offdiagonal_coefficient() const { return get_inv_dx_sq(); }

  // vertical adjacent offdiagonals: inv_dy_sq
  T get_vertical_offdiagonal_coefficent() const { return get_inv_dy_sq(); }

 public:
  const int Nx;  // number of internal points in x direction
  const int Ny;  // number of internal points in y direction

  // constructor
  Poisson(const poisson_config &cfg, const int seed)
      : Nx(cfg.X_res - 2), Ny(cfg.Y_res - 2), gen(seed) {}

  // initial state (all ones)
  std::vector<T> initialize_state() const {
    std::vector<T> state(get_state_dim(), static_cast<T>(1.0));
    return state;
  }

  // rhs = -alpha * state, alpha ~ U(0, 1)
  std::vector<T> eval_rhs(const std::vector<T> &state) const {
    // std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // const T alpha = static_cast<T>(uniform(gen));
    // std::vector<T> rhs(state.size());
    // for (int i = 0; i < static_cast<int>(state.size()); ++i) {
    //   rhs[i] = -alpha * state[i];
    // }
    std::vector<T> rhs(get_state_dim(), static_cast<T>(0.0));
    return rhs;
  }

  // coeff matrix
  std::vector<T> get_coefficient_matrix() const {
    // coeffients
    T diag_coeff = get_diagonal_coefficient();
    T horizontal_offdiag_coeff = get_horizontal_offdiagonal_coefficient();
    T vertical_offdiag_coeff = get_vertical_offdiagonal_coefficent();
    // initialize
    const int state_dim = get_state_dim();
    int diag_idx;
    std::vector<T> coeff(state_dim * state_dim, static_cast<T>(0.0));
    // allocate
    for (int i = 0; i < state_dim; i++) {
      // diagonal idx
      diag_idx = i * state_dim + i;
      // diag
      coeff[diag_idx] = diag_coeff;
      // horizontal offdiag (sub)
      if (i > 0) coeff[diag_idx - 1] = horizontal_offdiag_coeff;
      // horizontal offdiag (sup)
      if (i < state_dim - 1) coeff[diag_idx + 1] = horizontal_offdiag_coeff;
      // vertical offdiag (sub)
      if (i > Nx - 1) coeff[diag_idx - Nx] = vertical_offdiag_coeff;
      // vertical offdiag (sup)
      if (i < state_dim - Nx) coeff[diag_idx + Nx] = vertical_offdiag_coeff;
    }

    // enforce left boundary
    for (int i = Nx; i < state_dim; i += Nx) {
      diag_idx = i * state_dim + i;
      coeff[diag_idx - 1] = static_cast<T>(0.0);
    }

    // enforce right boundary
    for (int i = Nx - 1; i < state_dim; i += Nx) {
      diag_idx = i * state_dim + i;
      coeff[diag_idx + 1] = static_cast<T>(0.0);
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

 private:
  mutable std::mt19937 gen;  // seeded once at construction
};

/* poisson equation experiments
 * 1. runs Jacobi in a given precision
 * 2. runs block jacobi in mixed precision (single + half)
 */
void run_poisson_equation_experiments(Precision prec) {
  poisson_config cfg;
  cfg.prec = prec;
  Poisson<double> poisson(cfg, /*seed=*/42);
  std::vector<double> s = poisson.initialize_state();
  std::vector<double> r = poisson.eval_rhs(s);
  poisson.print_coefficient_matrix();
}
