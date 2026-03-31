#include <cuda_fp16.h>

#include <iomanip>
// #include <iostream>

#include "cholesky.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

// Compute L * L^T and return as std::vector<double>
template <typename T>
std::vector<double> compute_llt(const int N, std::vector<T> &h_l) {
  std::vector<double> llt(N * N, 0.0);
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      double val = 0.0;
      for (int k = 0; k < N; k++) {
        val += static_cast<double>(h_l[r * N + k]) *
               static_cast<double>(h_l[c * N + k]);
      }
      llt[r * N + c] = val;
    }
  }
  return llt;
}

// Compute Frobenius norm of A - L*L^T
template <typename T>
void compute_cholesky_error(const int N, std::vector<T> &h_a,
                            std::vector<T> &h_l) {
  double error = 0.0;
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      double llt_rc = 0.0;
      for (int k = 0; k < N; k++) {
        llt_rc += static_cast<double>(h_l[r * N + k]) *
                  static_cast<double>(h_l[c * N + k]);
      }
      double diff = static_cast<double>(h_a[r * N + c]) - llt_rc;
      error += diff * diff;
    }
  }
  std::cout << "Cholesky error (Frobenius norm): " << std::sqrt(error)
            << std::endl;
}

// Compute L * L^T and print the result
template <typename T>
void compute_and_print_llt(const int N, std::vector<T> &h_l) {
  std::cout << "L * L^T:\n";
  std::cout << std::scientific << std::setprecision(4);
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      double val = 0.0;
      for (int k = 0; k < N; k++) {
        val += static_cast<double>(h_l[r * N + k]) *
               static_cast<double>(h_l[c * N + k]);
      }
      std::cout << std::setw(14) << val;
    }
    std::cout << "\n";
  }
}

// Compute Vanilla Cholesky (Not Parallalizable)
void compute_vanilla_cholesky(const int N, std::vector<double> &h_a,
                              std::vector<double> &h_l) {
  for (int i = 0; i < N; i++) {
    // Compute the diagonals L_ii
    h_l[i * N + i] = h_a[i * N + i];

    double sum = 0.0;
    for (int k = 0; k < i; k++) {
      sum += h_l[i * N + k] * h_l[i * N + k];
    }
    h_l[i * N + i] -= sum;
    h_l[i * N + i] = std::sqrt(h_l[i * N + i]);

    // Compute the off-diagonals L_ji for all j > i and j < N
    for (int j = i + 1; j < N; j++) {
      sum = 0.0;
      for (int k = 0; k < i; k++) {
        sum += h_l[j * N + k] * h_l[i * N + k];
      }
      h_l[j * N + i] = (h_a[j * N + i] - sum) / h_l[i * N + i];
    }
  }
}

// Normalize Choleky kernel: A[row, row] = sqrt(A[row, row])
template <typename T>
__global__ void launch_cholesky_normalize_diagonal_kernel(const int N, T *a,
                                                          const int row,
                                                          Precision prec) {
  if (prec == Double) {
    a[row * N + row] = __dsqrt_rn(a[row * N + row]);
  } else if (prec == Single) {
    a[row * N + row] = __fsqrt_rn(a[row * N + row]);
  } else if (prec == Half) {
    a[row * N + row] = hsqrt(a[row * N + row]);
  } else {
    printf("<Cuda Error>: Invalid precision\n");
  }
}

// Normalize Choleky kernel: A[row, row] = sqrt(A[row, row])
template <typename T>
__global__ void launch_cholesky_scale_col_kernel(const int N, T *a,
                                                 const int row,
                                                 Precision prec) {
  // gid: effective row for the entry
  int gid = blockIdx.x * blockDim.x + threadIdx.x + row + 1;
  // scale
  if (gid < N) {
    if (prec == Double) {
      a[gid * N + row] = __ddiv_rn(a[gid * N + row], a[row * N + row]);

    } else if (prec == Single) {
      a[gid * N + row] = __fdiv_rn(a[gid * N + row], a[row * N + row]);

    } else if (prec == Half) {
      a[gid * N + row] = __hdiv(a[gid * N + row], a[row * N + row]);

    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

// Rank-1 update for Cholesky: A[i,j] -= A[i,row] * A[j,row]  for i >= j > row
template <typename T>
__global__ void launch_rank_one_update_kernel(const int N, T *a, const int row,
                                              Precision prec) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + row + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + row + 1;

  if ((i < N) && (j < N)) {
    if (prec == Double) {
      a[i * N + j] = __dsub_rn(
          a[i * N + j], __dmul_rn(a[(row + 1 + (i - (row + 1))) * N + row],
                                  a[(row + 1 + (j - (row + 1))) * N + row]));
    } else if (prec == Single) {
      a[i * N + j] = __fsub_rn(
          a[i * N + j], __fmul_rn(a[(row + 1 + (i - (row + 1))) * N + row],
                                  a[(row + 1 + (j - (row + 1))) * N + row]));
    } else if (prec == Half) {
      a[i * N + j] = __hsub(a[i * N + j],
                            __hmul(a[(row + 1 + (i - (row + 1))) * N + row],
                                   a[(row + 1 + (j - (row + 1))) * N + row]));
    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

// Right-looking Cholesky
template <typename T>
void launch_cholesky_decomposition_kernel(const int N, std::vector<T> &h_a,
                                          std::vector<T> &h_l, Precision prec) {
  // initialize
  T *d_a;
  int size = N * N * sizeof(T);
  dim3 blockDim = 64;

  // allocate
  cudaCheck(cudaMalloc((void **)&d_a, size));
  // transfer Host to Device
  cudaCheck(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));

  // Choleksy
  for (int row = 0; row < N; row++) {
    int tail = N - row - 1;

    // normalize the diagonal
    launch_cholesky_normalize_diagonal_kernel<<<1, 1>>>(N, d_a, row, prec);
    cudaCheck(cudaGetLastError());

    if (tail > 0) {
      // scale the cols
      dim3 gridDim_cols = get_grid_dim(tail, blockDim.x);
      launch_cholesky_scale_col_kernel<<<gridDim_cols, blockDim>>>(N, d_a, row,
                                                                   prec);
      cudaCheck(cudaGetLastError());

      // rank one udpate
      dim3 blockDim_rank(16, 16);
      dim3 gridDim_rank(get_grid_dim(tail, blockDim_rank.x),
                        get_grid_dim(tail, blockDim_rank.y));
      launch_rank_one_update_kernel<<<gridDim_rank, blockDim_rank>>>(N, d_a,
                                                                     row, prec);
      cudaCheck(cudaGetLastError());
    }
  }

  // transfer Device to Host
  cudaCheck(cudaMemcpy(h_l.data(), d_a, size, cudaMemcpyDeviceToHost));

  // zero out upper triangle (d_a is lower-triangular; upper retains values from
  // A)
  for (int r = 0; r < N; r++) {
    for (int c = r + 1; c < N; c++) {
      h_l[r * N + c] = static_cast<T>(0.0);
    }
  }

  // Free
  cudaFree(d_a);
}

// Forward solve kernel: solves Ly = b for y (single-thread, sequential)
template <typename T>
__global__ void forward_solve_kernel(const int N, const T *l, const T *b, T *y,
                                     Precision prec) {
  for (int i = 0; i < N; i++) {
    T sum = static_cast<T>(0.0);

    if (prec == Double) {
      for (int k = 0; k < i; k++)
        sum = __dadd_rn(sum, __dmul_rn(l[i * N + k], y[k]));

      y[i] = __ddiv_rn(__dsub_rn(b[i], sum), l[i * N + i]);

    } else if (prec == Single) {
      for (int k = 0; k < i; k++)
        sum = __fadd_rn(sum, __fmul_rn(l[i * N + k], y[k]));

      y[i] = __fdiv_rn(__fsub_rn(b[i], sum), l[i * N + i]);

    } else if (prec == Half) {
      for (int k = 0; k < i; k++)
        sum = __hadd_rn(sum, __hmul_rn(l[i * N + k], y[k]));

      y[i] = __hdiv(__hsub_rn(b[i], sum), l[i * N + i]);

    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

// Forward solve host launcher: solves Ly = b, result written to h_y
template <typename T>
void launch_forward_solve_kernel(const int N, const std::vector<T> &h_l,
                                 const std::vector<T> &h_b, std::vector<T> &h_y,
                                 Precision prec) {
  T *d_l, *d_b, *d_y;
  int size = N * sizeof(T);
  int size_mat = N * N * sizeof(T);
  // allocate
  cudaCheck(cudaMalloc((void **)&d_l, size_mat));
  cudaCheck(cudaMalloc((void **)&d_b, size));
  cudaCheck(cudaMalloc((void **)&d_y, size));
  // transfer
  cudaCheck(cudaMemcpy(d_l, h_l.data(), size_mat, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
  // cudaCheck(cudaMemset(d_y, 0, size));
  // launch single-thread sequential kernel
  forward_solve_kernel<<<1, 1>>>(N, d_l, d_b, d_y, prec);
  cudaCheck(cudaGetLastError());
  // copy result back
  cudaCheck(cudaMemcpy(h_y.data(), d_y, size, cudaMemcpyDeviceToHost));
  // free
  cudaFree(d_l);
  cudaFree(d_b);
  cudaFree(d_y);
}

// Backward solve kernel: solves L^T x = y for x (single-thread, sequential)
// L^T[i,k] = L[k,i], so x[i] = (y[i] - sum_{k=i+1}^{N-1} L[k,i]*x[k]) / L[i,i]
template <typename T>
__global__ void backward_solve_kernel(const int N, const T *l, const T *y, T *x,
                                      Precision prec) {
  for (int i = N - 1; i >= 0; i--) {
    T sum = static_cast<T>(0.0);

    if (prec == Double) {
      for (int k = i + 1; k < N; k++)
        sum = __dadd_rn(sum, __dmul_rn(l[k * N + i], x[k]));

      x[i] = __ddiv_rn(__dsub_rn(y[i], sum), l[i * N + i]);

    } else if (prec == Single) {
      for (int k = i + 1; k < N; k++)
        sum = __fadd_rn(sum, __fmul_rn(l[k * N + i], x[k]));

      x[i] = __fdiv_rn(__fsub_rn(y[i], sum), l[i * N + i]);

    } else if (prec == Half) {
      for (int k = i + 1; k < N; k++)
        sum = __hadd_rn(sum, __hmul_rn(l[k * N + i], x[k]));

      x[i] = __hdiv(__hsub_rn(y[i], sum), l[i * N + i]);

    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
  }
}

// Backward solve host launcher: solves L^T x = y, result written to h_x
template <typename T>
void launch_backward_solve_kernel(const int N, const std::vector<T> &h_l,
                                  const std::vector<T> &h_y,
                                  std::vector<T> &h_x, Precision prec) {
  T *d_l, *d_y, *d_x;
  int size = N * sizeof(T);
  int size_mat = N * N * sizeof(T);
  // allocate
  cudaCheck(cudaMalloc((void **)&d_l, size_mat));
  cudaCheck(cudaMalloc((void **)&d_y, size));
  cudaCheck(cudaMalloc((void **)&d_x, size));
  // transfer
  cudaCheck(cudaMemcpy(d_l, h_l.data(), size_mat, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_y, h_y.data(), size, cudaMemcpyHostToDevice));
  // cudaCheck(cudaMemset(d_x, 0, size));
  // launch single-thread sequential kernel
  backward_solve_kernel<<<1, 1>>>(N, d_l, d_y, d_x, prec);
  cudaCheck(cudaGetLastError());
  // copy result back
  cudaCheck(cudaMemcpy(h_x.data(), d_x, size, cudaMemcpyDeviceToHost));
  // free
  cudaFree(d_l);
  cudaFree(d_y);
  cudaFree(d_x);
}

// Cholesky solve: L L^T x = rhs
template <typename T>
void launch_cholesky_solve_kernel(const int N, const std::vector<T> &h_l,
                                  const std::vector<T> &h_rhs,
                                  std::vector<T> &h_result, Precision prec) {
  std::vector<T> h_y(N, static_cast<T>(0.0));
  // forward solve: Ly = rhs
  launch_forward_solve_kernel(N, h_l, h_rhs, h_y, prec);
  // backward solve: L^T x = y
  launch_backward_solve_kernel(N, h_l, h_y, h_result, prec);
}

template void launch_backward_solve_kernel<double>(const int,
                                                   const std::vector<double> &,
                                                   const std::vector<double> &,
                                                   std::vector<double> &,
                                                   Precision);
template void launch_backward_solve_kernel<float>(const int,
                                                  const std::vector<float> &,
                                                  const std::vector<float> &,
                                                  std::vector<float> &,
                                                  Precision);
template void launch_backward_solve_kernel<half>(const int,
                                                 const std::vector<half> &,
                                                 const std::vector<half> &,
                                                 std::vector<half> &,
                                                 Precision);

template void launch_forward_solve_kernel<double>(const int,
                                                  const std::vector<double> &,
                                                  const std::vector<double> &,
                                                  std::vector<double> &,
                                                  Precision);
template void launch_forward_solve_kernel<float>(const int,
                                                 const std::vector<float> &,
                                                 const std::vector<float> &,
                                                 std::vector<float> &,
                                                 Precision);
template void launch_forward_solve_kernel<half>(const int,
                                                const std::vector<half> &,
                                                const std::vector<half> &,
                                                std::vector<half> &, Precision);

template void launch_cholesky_solve_kernel<double>(const int,
                                                   const std::vector<double> &,
                                                   const std::vector<double> &,
                                                   std::vector<double> &,
                                                   Precision);
template void launch_cholesky_solve_kernel<float>(const int,
                                                  const std::vector<float> &,
                                                  const std::vector<float> &,
                                                  std::vector<float> &,
                                                  Precision);
template void launch_cholesky_solve_kernel<half>(const int,
                                                 const std::vector<half> &,
                                                 const std::vector<half> &,
                                                 std::vector<half> &,
                                                 Precision);

template void compute_cholesky_error<double>(const int, std::vector<double> &,
                                             std::vector<double> &);
template void compute_cholesky_error<float>(const int, std::vector<float> &,
                                            std::vector<float> &);
template void compute_cholesky_error<half>(const int, std::vector<half> &,
                                           std::vector<half> &);

template std::vector<double> compute_llt<double>(const int,
                                                 std::vector<double> &);
template std::vector<double> compute_llt<float>(const int,
                                                std::vector<float> &);
template std::vector<double> compute_llt<half>(const int, std::vector<half> &);

template void compute_and_print_llt<double>(const int, std::vector<double> &);
template void compute_and_print_llt<float>(const int, std::vector<float> &);
template void compute_and_print_llt<half>(const int, std::vector<half> &);

template void launch_cholesky_decomposition_kernel<double>(
    const int, std::vector<double> &, std::vector<double> &, Precision);
template void launch_cholesky_decomposition_kernel<float>(const int,
                                                          std::vector<float> &,
                                                          std::vector<float> &,
                                                          Precision);
template void launch_cholesky_decomposition_kernel<half>(const int,
                                                         std::vector<half> &,
                                                         std::vector<half> &,
                                                         Precision);
