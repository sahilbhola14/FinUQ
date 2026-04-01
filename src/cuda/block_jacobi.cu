#include <cuda_fp16.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "block_jacobi.cuh"
#include "cholesky.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

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

// rhs[gid] = b[gid] - (A*x)[gid], matvec done in tiles of blk_jacobi_tile_size
template <typename T>
__global__ void block_jacobi_rhs_kernel(const int N, const T *coeff,
                                        const T *state, const T *b, T *rhs,
                                        const int tile_size, Precision prec) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    const int num_tiles = (N + tile_size - 1) / tile_size;

    T dot = static_cast<T>(0.0);

    if (prec == Double) {
      // loop over all tiles
      for (int t = 0; t < num_tiles; t++) {
        // tile start
        const int tile_start = t * tile_size;
        // loop within the tile (tile_start + k) < N ensures safeguard
        for (int k = 0; k < tile_size && (tile_start + k) < N; k++)
          dot = __dadd_rn(dot, __dmul_rn(coeff[gid * N + tile_start + k],
                                         state[tile_start + k]));
      }
      // update the rhs
      rhs[gid] = __dsub_rn(b[gid], dot);

    } else if (prec == Single) {
      for (int t = 0; t < num_tiles; t++) {
        const int tile_start = t * tile_size;

        for (int k = 0; k < tile_size && (tile_start + k) < N; k++)
          dot = __fadd_rn(dot, __fmul_rn(coeff[gid * N + tile_start + k],
                                         state[tile_start + k]));
      }

      rhs[gid] = __fsub_rn(b[gid], dot);

    } else if (prec == Half) {
      for (int t = 0; t < num_tiles; t++) {
        const int tile_start = t * tile_size;
        for (int k = 0; k < tile_size && (tile_start + k) < N; k++)
          dot = __hadd_rn(dot, __hmul_rn(coeff[gid * N + tile_start + k],
                                         state[tile_start + k]));
      }

      rhs[gid] = __hsub_rn(b[gid], dot);

    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

// block jacobi update step kernel
template <typename T>
__global__ void block_jacobi_update_step_kernel(const int num_tiles,
                                                const int tile_size,
                                                const T *chol, const T *rhs,
                                                const T *state, T *state_new,
                                                Precision prec) {
  // initialzie
  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *y = smem + threadIdx.x * 2 * tile_size;  // begining of y sol
  T *c = y + tile_size;                       // begining of the correction c

  const int t = blockIdx.x * blockDim.x + threadIdx.x;  // tile idx
  if (t >= num_tiles) return;                           // boundary check

  const int tile_start = t * tile_size;  // basically global row start
  const T *L =
      chol + t * tile_size * tile_size;  // start of the cholesky factor

  // forward solve: L * y = rhs_tile
  for (int i = 0; i < tile_size; i++) {
    T sum = static_cast<T>(0.0);
    if (prec == Double) {
      for (int j = 0; j < i; j++)
        sum = __dadd_rn(sum, __dmul_rn(L[i * tile_size + j], y[j]));
      y[i] =
          __ddiv_rn(__dsub_rn(rhs[tile_start + i], sum), L[i * tile_size + i]);
    } else if (prec == Single) {
      for (int j = 0; j < i; j++)
        sum = __fadd_rn(sum, __fmul_rn(L[i * tile_size + j], y[j]));
      y[i] =
          __fdiv_rn(__fsub_rn(rhs[tile_start + i], sum), L[i * tile_size + i]);
    } else if (prec == Half) {
      for (int j = 0; j < i; j++)
        sum = __hadd_rn(sum, __hmul_rn(L[i * tile_size + j], y[j]));
      y[i] = __hdiv(__hsub_rn(rhs[tile_start + i], sum), L[i * tile_size + i]);
    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }

  // backward solve: L^T * c = y
  for (int i = tile_size - 1; i >= 0; i--) {
    T sum = static_cast<T>(0.0);
    if (prec == Double) {
      for (int j = i + 1; j < tile_size; j++)
        sum = __dadd_rn(sum, __dmul_rn(L[j * tile_size + i], c[j]));
      c[i] = __ddiv_rn(__dsub_rn(y[i], sum), L[i * tile_size + i]);
    } else if (prec == Single) {
      for (int j = i + 1; j < tile_size; j++)
        sum = __fadd_rn(sum, __fmul_rn(L[j * tile_size + i], c[j]));
      c[i] = __fdiv_rn(__fsub_rn(y[i], sum), L[i * tile_size + i]);
    } else if (prec == Half) {
      for (int j = i + 1; j < tile_size; j++)
        sum = __hadd_rn(sum, __hmul_rn(L[j * tile_size + i], c[j]));
      c[i] = __hdiv(__hsub_rn(y[i], sum), L[i * tile_size + i]);
    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }

  // state update: state_new = state + c
  for (int k = 0; k < tile_size; k++) {
    if (prec == Double) {
      state_new[tile_start + k] = __dadd_rn(state[tile_start + k], c[k]);
    } else if (prec == Single) {
      state_new[tile_start + k] = __fadd_rn(state[tile_start + k], c[k]);
    } else if (prec == Half) {
      state_new[tile_start + k] = __hadd_rn(state[tile_start + k], c[k]);
    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

// compute b - A*x^k for the rhs of the update form of block jacobi
template <typename T>
void launch_block_jacobi_rhs(const int N, const std::vector<T> &h_coeff,
                             const std::vector<T> &h_state,
                             const std::vector<T> &h_b, std::vector<T> &h_rhs,
                             const int blk_jacobi_matvect_tile_size,
                             Precision prec) {
  // initialize
  const int size = N * sizeof(T);
  const int size_mat = N * N * sizeof(T);
  T *d_coeff, *d_state, *d_b, *d_rhs;
  // memory allocation
  cudaCheck(cudaMalloc((void **)&d_coeff, size_mat));
  cudaCheck(cudaMalloc((void **)&d_state, size));
  cudaCheck(cudaMalloc((void **)&d_b, size));
  cudaCheck(cudaMalloc((void **)&d_rhs, size));
  // transfer: host to device
  cudaCheck(
      cudaMemcpy(d_coeff, h_coeff.data(), size_mat, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_state, h_state.data(), size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
  // kernel parameters
  dim3 blockDim = 64;
  dim3 gridDim = get_grid_dim(N, blockDim.x);
  block_jacobi_rhs_kernel<<<gridDim, blockDim>>>(
      N, d_coeff, d_state, d_b, d_rhs, blk_jacobi_matvect_tile_size, prec);
  cudaCheck(cudaGetLastError());
  // transfer: device to host
  cudaCheck(cudaMemcpy(h_rhs.data(), d_rhs, size, cudaMemcpyDeviceToHost));
  // Free
  cudaFree(d_coeff);
  cudaFree(d_state);
  cudaFree(d_b);
  cudaFree(d_rhs);
}

// Block Jacobi update step
// 1. perform the linear system solve for small blocks
// 2. update x^k+1 = x^k + c, where c is the solution from the linear system
// solve
template <typename T>
void launch_block_jacobi_sequential_update_step(
    const int N, const std::vector<Matrix<T>> &cholesky_factors,
    const std::vector<T> &h_rhs, const std::vector<T> &h_state,
    std::vector<T> &h_state_new, const int blk_jacobi_tile_size,
    Precision prec) {
  // #TODO: do parallel solve
  const int num_tiles = static_cast<int>(cholesky_factors.size());
  const int tile_size = blk_jacobi_tile_size;

  for (int t = 0; t < num_tiles; t++) {
    const int tile_start = t * tile_size;

    // extract rhs tile
    std::vector<T> rhs_tile(tile_size);
    for (int k = 0; k < tile_size; k++) rhs_tile[k] = h_rhs[tile_start + k];

    // solve: L * L^T * c = rhs_tile
    std::vector<T> c_tile(tile_size, static_cast<T>(0.0));
    launch_cholesky_solve_kernel(tile_size, cholesky_factors[t].data, rhs_tile,
                                 c_tile, prec);

    // update: x^{k+1} = x^k + c
    for (int k = 0; k < tile_size; k++)
      h_state_new[tile_start + k] =
          static_cast<T>(static_cast<double>(h_state[tile_start + k]) +
                         static_cast<double>(c_tile[k]));
  }
}

// Block Jacobi update step
// 1. perform the linear system solve for small blocks
// 2. update x^k+1 = x^k + c, where c is the solution from the linear system
// solve
template <typename T>
void launch_block_jacobi_update_step(
    const int N, const std::vector<Matrix<T>> &cholesky_factors,
    const std::vector<T> &h_rhs, const std::vector<T> &h_state,
    std::vector<T> &h_state_new, const int blk_jacobi_tile_size,
    Precision prec) {
  // initialization
  const int num_tiles = cholesky_factors.size();
  const int tile_size = blk_jacobi_tile_size;
  const int chol_stride = tile_size * tile_size;
  T *d_chol, *d_rhs, *d_state, *d_state_new;

  // flatten cholesky factors into continuous host array
  std::vector<T> h_chol_flat(num_tiles * chol_stride);
  for (int t = 0; t < num_tiles; t++) {
    std::copy(cholesky_factors[t].data.begin(), cholesky_factors[t].data.end(),
              h_chol_flat.begin() + t * chol_stride);
  }
  // memory allocattion
  cudaCheck(cudaMalloc((void **)&d_chol, num_tiles * chol_stride * sizeof(T)));
  cudaCheck(cudaMalloc((void **)&d_rhs, N * sizeof(T)));
  cudaCheck(cudaMalloc((void **)&d_state, N * sizeof(T)));
  cudaCheck(cudaMalloc((void **)&d_state_new, N * sizeof(T)));
  // host to device
  cudaCheck(cudaMemcpy(d_chol, h_chol_flat.data(),
                       num_tiles * chol_stride * sizeof(T),
                       cudaMemcpyHostToDevice));
  cudaCheck(
      cudaMemcpy(d_rhs, h_rhs.data(), N * sizeof(T), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_state, h_state.data(), N * sizeof(T),
                       cudaMemcpyHostToDevice));

  // kernel parameters
  dim3 blockDim = 64;
  dim3 gridDim =
      get_grid_dim(num_tiles, blockDim.x);  // each thread work on one block
  // each thread has its own 2 * tile_size
  size_t smem_size = blockDim.x * 2 * tile_size * sizeof(T);
  // kernel launch
  block_jacobi_update_step_kernel<<<gridDim, blockDim, smem_size>>>(
      num_tiles, tile_size, d_chol, d_rhs, d_state, d_state_new, prec);
  cudaCheck(cudaGetLastError());
  // device -> host
  cudaCheck(cudaMemcpy(h_state_new.data(), d_state_new, N * sizeof(T),
                       cudaMemcpyDeviceToHost));

  // Free
  cudaFree(d_chol);
  cudaFree(d_rhs);
  cudaFree(d_state);
  cudaFree(d_state_new);
}

// Block Jacobi (single iteration)
// Solve Ax = b using the update form.
// Dc = b - Ax^k, where D is the diagonal block.
// x^{k+1} = x^k + c
template <typename T>
void launch_block_jacobi_solver_single_step(
    const std::vector<T> &h_coeff,
    const std::vector<Matrix<T>> &cholesky_factors, std::vector<T> &h_state,
    std::vector<T> &h_state_new, const poisson_rhs_config<T> &poisson_rhs_cfg,
    double &h_error, const poisson_config &poisson_cfg, bool verbose = false) {
  // initialize
  const int state_dim = poisson_rhs_cfg.state_dim;

  // (1) evaluate b
  std::vector<T> h_b(state_dim);
  eval_rhs(poisson_rhs_cfg, h_state.data(), h_b.data());
  // (2) b - Ax^k
  std::vector<T> h_rhs(state_dim, static_cast<T>(0.0));
  launch_block_jacobi_rhs(state_dim, h_coeff, h_state, h_b, h_rhs,
                          poisson_cfg.blk_jacobi_matvect_tile_size,
                          poisson_cfg.prec);
  // (3): block jacobi solve
  launch_block_jacobi_update_step(state_dim, cholesky_factors, h_rhs, h_state,
                                  h_state_new, poisson_cfg.blk_jacobi_tile_size,
                                  poisson_cfg.prec);

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

// Block Jacobi Solvert
template <typename T>
void launch_block_jacobi_solver(const poisson_rhs_config<T> &poisson_rhs_cfg,
                                std::vector<T> &h_coeff,
                                std::vector<T> &h_state,
                                const poisson_config &poisson_cfg,
                                bool verbose) {
  // initializations
  const double etol = poisson_cfg.etol;
  const int max_iter = poisson_cfg.max_iter;
  const int state_dim = poisson_rhs_cfg.state_dim;
  std::vector<T> h_state_new(state_dim);
  double h_error = 2.0 * etol;  // ensure at least one iteration

  // (1): compute the cholesky factors per tile in precision: prec_cholesky
  std::vector<Matrix<T>> cholesky_factors =
      compute_cholesky_per_jacobi_tile(h_coeff, poisson_cfg);

  // (2): block jacobi iteration
  int k = 0;
  for (; k < max_iter && h_error > etol; k++) {
    launch_block_jacobi_solver_single_step(h_coeff, cholesky_factors, h_state,
                                           h_state_new, poisson_rhs_cfg,
                                           h_error, poisson_cfg);

    if (verbose && k % 10 == 0) {
      printf("Iter %d | error: %e\n", k, h_error);
    }

    std::swap(h_state, h_state_new);
  }

  // save solution to csv: columns = i_x, i_y, value
  const int Nx = poisson_rhs_cfg.Nx;
  // const int Ny = state_dim / Nx;
  std::ostringstream ss;
  ss << "poisson_solution_" << to_string(poisson_cfg.prec) << "_prec"
     << "_chol_" << to_string(poisson_cfg.prec_cholesky) << "_zeta_"
     << std::fixed << std::setprecision(6)
     << static_cast<double>(poisson_rhs_cfg.zeta) << ".csv";
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

// template initialization
template void launch_block_jacobi_solver<double>(
    const poisson_rhs_config<double> &, std::vector<double> &,
    std::vector<double> &, const poisson_config &, bool);
template void launch_block_jacobi_solver<float>(
    const poisson_rhs_config<float> &, std::vector<float> &,
    std::vector<float> &, const poisson_config &, bool);
template void launch_block_jacobi_solver<half>(const poisson_rhs_config<half> &,
                                               std::vector<half> &,
                                               std::vector<half> &,
                                               const poisson_config &, bool);
