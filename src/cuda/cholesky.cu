#include <cuda_fp16.h>

#include "cholesky.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

// Compute Vanilla Cholesky (Not Parallalizable)
void compute_vanilla_cholesky(const int N, double *a, double *l) {
  for (int i = 0; i < N; i++) {
    // Compute the diagonals L_ii
    l[i * N + i] = a[i * N + i];

    double sum = 0.0;
    for (int k = 0; k < i; k++) {
      sum += l[i * N + k] * l[i * N + k];
    }
    l[i * N + i] -= sum;
    l[i * N + i] = std::sqrt(l[i * N + i]);

    // Compute the off-diagonals L_ji for all j > i and j < N
    for (int j = i + 1; j < N; j++) {
      sum = 0.0;
      for (int k = 0; k < i; k++) {
        sum += l[j * N + k] * l[i * N + k];
      }
      l[j * N + i] = (a[j * N + i] - sum) / l[i * N + i];
    }
  }
}

// Normalize Choleky kernel: A[row, row] = sqrt(A[row, row])
template <typename T>
__global__ void launch_cholesky_normalize_diagonal_kernel(const int N, T *a,
                                                          Precision prec) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < N) {
    if (prec == Double) {
      a[gid * N + gid] = __dsqrt_rn(a[gid * N + gid]);
    } else if (prec == Single) {
      a[gid * N + gid] = __fsqrt_rn(a[gid * N + gid]);
    } else if (prec == Half) {
      a[gid * N + gid] = hsqrt(a[gid * N + gid]);
    } else {
      printf("<Cuda Error>: Invalid precision\n");
    }
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

template <typename T>
void launch_cholesky_decomposition_kernel(const int N, std::vector<T> &h_a,
                                          std::vector<T> &h_l, Precision prec) {
  // initialize
  T *d_a, *d_l;
  int size = N * N * sizeof(T);
  dim3 blockDim = 64;
  // allocate
  cudaCheck(cudaMalloc((void **)&d_a, size));
  cudaCheck(cudaMalloc((void **)&d_l, size));
  // transfer
  cudaCheck(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
  // normalize the diagonals
  dim3 gridDim_diag =
      get_grid_dim(N, blockDim.x);  // threads mapped to diagonal
  launch_cholesky_normalize_diagonal_kernel<<<gridDim_diag, blockDim>>>(N, d_a,
                                                                        prec);
  cudaCheck(cudaGetLastError());

  // normalize the columns and perform rank one update
  for (int row = 0; row < N; row++) {
    // number of elmenets below the diag.
    int tail = N - row - 1;

    if (tail > 0) {
      // threads mapped to elements below the col
      dim3 gridDim_sub = get_grid_dim(tail, blockDim.x);
      launch_cholesky_scale_col_kernel<<<gridDim_sub, blockDim>>>(N, d_a, row,
                                                                  prec);
      cudaCheck(cudaGetLastError());
    }
  }

  // Free
  cudaFree(d_a);
  cudaFree(d_l);
}

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
