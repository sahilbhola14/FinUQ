#include <cuda_fp16.h>

#include <iostream>

#include "matrix_vector_cuda.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* matrix-vector product kernel */
template <typename T>
__global__ void matvec_product_kernel(const int rows, const int cols, T *matrix,
                                      T *a, T *result, Precision prec) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < rows) {
    T sum = static_cast<T>(0.0);
    for (int i = 0; i < cols; i++) {
      if (prec == Double) {
        sum = __dadd_rn(sum, __dmul_rn(matrix[gid * cols + i], a[i]));
      } else if (prec == Single) {
        sum = __fadd_rn(sum, __fmul_rn(matrix[gid * cols + i], a[i]));
      } else if (prec == Half) {
        sum =
            __hadd_rn(sum, __hmul_rn(static_cast<half>(matrix[gid * cols + i]),
                                     static_cast<half>(a[i])));
      }
    }

    /* copy */
    result[gid] = sum;
  }
}

/* matrix-vector product kernel launcher */
template <typename T>
void launch_matvec_product_kernel(const Matrix<T> &h_matrix,
                                  const std::vector<T> &h_a,
                                  std::vector<T> &h_result, Precision prec,
                                  bool verbose) {
  /* kernel parameters */
  dim3 blockDim = 256;
  dim3 gridDim = get_grid_dim(h_matrix.rows, blockDim.x);
  /* initialize */
  T *d_matrix, *d_a, *d_result;
  int matrix_size = h_matrix.rows * h_matrix.cols * sizeof(T);
  int a_size = h_matrix.cols * sizeof(T);
  int result_size = h_matrix.rows * sizeof(T);
  /* allocate memory */
  cudaCheck(cudaMalloc((void **)&d_matrix, matrix_size));
  cudaCheck(cudaMalloc((void **)&d_a, a_size));
  cudaCheck(cudaMalloc((void **)&d_result, result_size));
  /* host to device */
  cudaCheck(cudaMemcpy(d_matrix, h_matrix.data.data(), matrix_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_a, h_a.data(), a_size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for matrix-vector product in "
              << to_string(prec) << " precision" << std::endl;
  matvec_product_kernel<<<gridDim, blockDim>>>(h_matrix.rows, h_matrix.cols,
                                               d_matrix, d_a, d_result, prec);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(cudaMemcpy(h_result.data(), d_result, result_size,
                       cudaMemcpyDeviceToHost));
  /* free */
  cudaFree(d_matrix);
  cudaFree(d_a);
  cudaFree(d_result);
}

/* initialize template */
template void launch_matvec_product_kernel<double>(const Matrix<double> &,
                                                   const std::vector<double> &,
                                                   std::vector<double> &,
                                                   Precision, bool);
