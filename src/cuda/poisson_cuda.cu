#include <cuda_fp16.h>

#include "poisson.hpp"
#include "poisson_cuda.cuh"

template <typename T>
void launch_jacobi_solver_single_step(std::vector<T> &h_coeff,
                                      std::vector<T> &h_state,
                                      std::vector<T> &h_state_new,
                                      const poisson_rhs_config<T> &cfg,
                                      Precision prec, double &h_error) {}

template <typename T>
void launch_jacobi_solver(const poisson_rhs_config<T> &poisson_rhs_cfg,
                          std::vector<T> &h_coeff, std::vector<T> &h_state,
                          Precision prec, const double etol, const int max_iter,
                          bool verbose) {
  const int state_dim = poisson_rhs_cfg.state_dim;
  std::vector<T> h_state_new(state_dim);
  double h_error = 2.0 * etol;  // ensure at least one iteration

  for (int k = 1; k < max_iter && h_error > etol; k++) {
    launch_jacobi_solver_single_step(h_coeff, h_state, h_state_new,
                                     poisson_rhs_cfg, prec, h_error);
    if (verbose && k % 10 == 0) {
      printf("Iter %d | error: %e\n", k, h_error);
    }
    std::swap(h_state, h_state_new);
  }
}

// explicit instantiations
template void launch_jacobi_solver<double>(const poisson_rhs_config<double> &,
                                           std::vector<double> &,
                                           std::vector<double> &, Precision,
                                           const double, const int, bool);
template void launch_jacobi_solver<float>(const poisson_rhs_config<float> &,
                                          std::vector<float> &,
                                          std::vector<float> &, Precision,
                                          const double, const int, bool);
template void launch_jacobi_solver<half>(const poisson_rhs_config<half> &,
                                         std::vector<half> &,
                                         std::vector<half> &, Precision,
                                         const double, const int, bool);
