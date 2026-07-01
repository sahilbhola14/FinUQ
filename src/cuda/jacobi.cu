#include <cuda_fp16.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "jacobi.cuh"
#include "poisson.hpp"
#include "utils.hpp"
#include "utils_cuda.cuh"

template <typename T>
__global__ void launch_jacobi_solver_single_step_kernel(const int state_dim,
                                                        T *coeff, T *state,
                                                        T *rhs, T *state_new,
                                                        Precision prec) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  T Axk = static_cast<T>(0.0);

  if (gid < state_dim) {
    // compute Ax^k
    if (prec == Double) {
      for (int j = 0; j < state_dim; j++) {
        Axk = __dadd_rn(Axk, __dmul_rn(coeff[gid * state_dim + j], state[j]));
      }
      // x^{k+1} = x^k + (b - Ax^k) / D
      state_new[gid] = __dadd_rn(
          state[gid],
          __ddiv_rn(__dsub_rn(rhs[gid], Axk), coeff[gid * state_dim + gid]));

    } else if (prec == Single) {
      for (int j = 0; j < state_dim; j++) {
        Axk = __fadd_rn(Axk, __fmul_rn(coeff[gid * state_dim + j], state[j]));
      }
      state_new[gid] = __fadd_rn(
          state[gid],
          __fdiv_rn(__fsub_rn(rhs[gid], Axk), coeff[gid * state_dim + gid]));

    } else if (prec == Half) {
      for (int j = 0; j < state_dim; j++) {
        Axk = __hadd_rn(Axk, __hmul_rn(coeff[gid * state_dim + j], state[j]));
      }
      state_new[gid] = __hadd_rn(
          state[gid],
          __hdiv(__hsub_rn(rhs[gid], Axk), coeff[gid * state_dim + gid]));

    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

/* single step of the jacobi
 * 1. Solve Ax = b as D*x^{k+1} = b - A*x^k
 */
template <typename T>
void launch_jacobi_solver_single_step(
    std::vector<T> &h_coeff, std::vector<T> &h_state,
    std::vector<T> &h_state_new, std::vector<T> &h_rhs,
    const poisson_solver_config<T> &poisson_solver_cfg, double &h_error,
    Precision prec, bool verbose = false) {
  // initialize
  const int state_dim = poisson_solver_cfg.state_dim;
  int size = state_dim * sizeof(T);
  int size_mat = state_dim * state_dim * sizeof(T);
  T *d_coeff, *d_state, *d_state_new, *d_rhs;

  // memory allocation
  cudaCheck(cudaMalloc((void **)&d_coeff, size_mat));
  cudaCheck(cudaMalloc((void **)&d_state, size));
  cudaCheck(cudaMalloc((void **)&d_rhs, size));
  cudaCheck(cudaMalloc((void **)&d_state_new, size));

  // transfer
  cudaCheck(
      cudaMemcpy(d_coeff, h_coeff.data(), size_mat, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_state, h_state.data(), size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_rhs, h_rhs.data(), size, cudaMemcpyHostToDevice));

  // kernel parameters
  dim3 blockDim = 64;
  dim3 gridDim = get_grid_dim(state_dim, blockDim.x);
  // kernel launch
  if (verbose == true)
    std::cout << "launching Jacobi kernel (sweep form) in " << to_string(prec)
              << " precision" << std::endl;

  launch_jacobi_solver_single_step_kernel<<<gridDim, blockDim>>>(
      state_dim, d_coeff, d_state, d_rhs, d_state_new, prec);
  cudaCheck(cudaGetLastError());
  // copy Device to Host
  cudaCheck(cudaMemcpy(h_state_new.data(), d_state_new, size,
                       cudaMemcpyDeviceToHost));
  // Free
  cudaFree(d_coeff);
  cudaFree(d_state);
  cudaFree(d_rhs);
  cudaFree(d_state_new);

  // compute relative L2 error: ||x_new - x_old|| / ||x_old||
  double num = 0.0, denom = 0.0;
  for (int i = 0; i < state_dim; i++) {
    double diff =
        static_cast<double>(h_state_new[i]) - static_cast<double>(h_state[i]);
    num += diff * diff;
    denom += static_cast<double>(h_state[i]) * static_cast<double>(h_state[i]);
  }
  h_error = std::sqrt(num) / std::sqrt(denom);
}

template <typename T>
void launch_jacobi_solver(std::vector<T> &h_coeff, std::vector<T> &h_state,
                          std::vector<T> &h_rhs, Precision prec,
                          const double etol, const int max_iter,
                          const poisson_solver_config<T> &poisson_solver_cfg,
                          bool verbose) {
  const int state_dim = poisson_solver_cfg.state_dim;
  std::vector<T> h_state_new(state_dim);
  double h_error = 2.0 * etol;  // ensure at least one iteration

  int k = 0;
  for (; k < max_iter && h_error > etol; k++) {
    launch_jacobi_solver_single_step(h_coeff, h_state, h_state_new, h_rhs,
                                     poisson_solver_cfg, h_error, prec);
    if (verbose && k % 10 == 0) {
      printf("Iter %d | error: %e\n", k, h_error);
    }
    std::swap(h_state, h_state_new);
  }

  // Note: due to swapping, after the convergence: h_state is the final answer.

  // save solution to csv: columns = i_x, i_y, value
  const int Nx = poisson_solver_cfg.Nx;
  const int Ny = state_dim / Nx;
  std::ostringstream ss;
  ss << "poisson_solution_" << to_string(prec) << "_prec"
     << "_zeta_" << std::fixed << std::setprecision(6)
     << static_cast<double>(poisson_solver_cfg.zeta) << ".csv";
  std::ofstream file(ss.str());
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << ss.str() << "\n";
    return;
  }
  file << "i_x,i_y,value\n";
  for (int idx = 0; idx < state_dim; idx++) {
    int i_x = idx % Nx;
    int i_y = idx / Nx;
    file << i_x << "," << i_y << "," << std::scientific << std::setprecision(10)
         << static_cast<double>(h_state[idx]) << "\n";
  }
  file.close();
  if (verbose) {
    std::cout << "solution saved to " << ss.str() << " (converged in " << k
              << " iters, error=" << h_error << ")\n";
  }
}

// explicit instantiations
template void launch_jacobi_solver<double>(
    std::vector<double> &, std::vector<double> &, std::vector<double> &,
    Precision, const double, const int, const poisson_solver_config<double> &,
    bool);
template void launch_jacobi_solver<float>(
    std::vector<float> &, std::vector<float> &, std::vector<float> &, Precision,
    const double, const int, const poisson_solver_config<float> &, bool);
template void launch_jacobi_solver<half>(
    std::vector<half> &, std::vector<half> &, std::vector<half> &, Precision,
    const double, const int, const poisson_solver_config<half> &, bool);
