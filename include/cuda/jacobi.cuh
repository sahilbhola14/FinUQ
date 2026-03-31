#ifndef JACOBI_CUH
#define JACOBI_CUH

#include <vector>

#include "definition.hpp"

// allow __host__ __device__ to compile under g++ as well
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// parameters needed to evaluate the state-dependent rhs on host or device
template <typename T>
struct poisson_rhs_config {
  T zeta;         // rhs scaling factor
  T inv_dy_sq;    // top boundary condition coefficient
  int Nx;         // number of internal x-points (extent of top BC)
  int state_dim;  // total state dimension (Nx * Ny)
};

// rhs[i] = -zeta * state[i], with top boundary condition applied for i < Nx
template <typename T>
__host__ __device__ void eval_rhs(const poisson_rhs_config<T> &cfg,
                                  const T *state, T *rhs) {
  // original rhs
  for (int i = 0; i < cfg.state_dim; i++) {
    // rhs[i] = -cfg.zeta * state[i];
    rhs[i] = static_cast<T>(0.0);
  }
  // apply top boundary condition
  for (int i = 0; i < cfg.Nx; i++) {
    rhs[i] += cfg.inv_dy_sq;
  }
}

// Jacobi solve
template <typename T>
void launch_jacobi_solver(const poisson_rhs_config<T> &poisson_rhs_cfg,
                          std::vector<T> &h_coeff,
                          std::vector<T> &h_state_initial, Precision prec,
                          const double etol, const int max_iter,
                          bool verbose = false);

#endif
