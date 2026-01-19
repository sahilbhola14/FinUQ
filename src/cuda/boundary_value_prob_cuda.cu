#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <iostream>

#include "boundary_value_prob_cuda.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* LU decomposition kernel */
template <typename T>
__global__ void lu_decomposition_kernel(const int n, T *sub_diag, T *main_diag,
                                        T *super_diag, T *l_sub_diag,
                                        T *u_main_diag) {
  /* initialize */
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /* compute the LU decomposition */
  u_main_diag[0] = main_diag[0];
  for (int i = 1; i < n; i++) {
    l_sub_diag[i] = sub_diag[i] / u_main_diag[i - 1];
    u_main_diag[i] = main_diag[i] - l_sub_diag[i] * super_diag[i - 1];
  }
}

/* LU decomposition kernel launcher */
template <typename T>
void launch_lu_decomposition_kernel(const int num_intervals,
                                    const std::vector<T> &h_sub_diag,
                                    const std::vector<T> &h_main_diag,
                                    const std::vector<T> &h_super_diag,
                                    std::vector<T> &h_l_sub_diag,
                                    std::vector<T> &h_u_main_diag,
                                    Precision prec, bool verbose = false) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  int size = Ns * sizeof(T);
  T *d_sub_diag, *d_main_diag, *d_super_diag, *d_l_sub_diag, *d_u_main_diag;
  /* allocate memory */
  cudaCheck(cudaMalloc((void **)&d_sub_diag, size));
  cudaCheck(cudaMalloc((void **)&d_main_diag, size));
  cudaCheck(cudaMalloc((void **)&d_super_diag, size));
  cudaCheck(cudaMalloc((void **)&d_l_sub_diag, size));
  cudaCheck(cudaMalloc((void **)&d_u_main_diag, size));
  /* host to device */
  cudaCheck(
      cudaMemcpy(d_sub_diag, h_sub_diag.data(), size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_main_diag, h_main_diag.data(), size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_super_diag, h_super_diag.data(), size,
                       cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for LU decomposition in " << to_string(prec)
              << " precision" << std::endl;

  lu_decomposition_kernel<<<gridDim, blockDim>>>(
      Ns, d_sub_diag, d_main_diag, d_super_diag, d_l_sub_diag, d_u_main_diag);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(cudaMemcpy(h_l_sub_diag.data(), d_l_sub_diag, size,
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_u_main_diag.data(), d_u_main_diag, size,
                       cudaMemcpyDeviceToHost));
  /* free */
  cudaCheck(cudaFree(d_sub_diag));
  cudaCheck(cudaFree(d_main_diag));
  cudaCheck(cudaFree(d_super_diag));
  cudaCheck(cudaFree(d_l_sub_diag));
  cudaCheck(cudaFree(d_u_main_diag));
}

template <typename T>
void launch_thomas_algorithm_kernel(const int num_intervals,
                                    const std::vector<T> &h_sub_diag,
                                    const std::vector<T> &h_main_diag,
                                    const std::vector<T> &h_super_diag,
                                    const std::vector<T> &h_rhs,
                                    std::vector<T> &h_state, Precision prec) {
  // initialize
  const int Ns = num_intervals - 1;
  std::vector<T> h_l_sub_diag(Ns), h_u_main_diag(Ns);

  // LU decomposition
  launch_lu_decomposition_kernel<T>(num_intervals, h_sub_diag, h_main_diag,
                                    h_super_diag, h_l_sub_diag, h_u_main_diag,
                                    prec);
}

/* template initialization */
template void launch_thomas_algorithm_kernel<double>(
    const int, const std::vector<double> &, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    std::vector<double> &, Precision);
