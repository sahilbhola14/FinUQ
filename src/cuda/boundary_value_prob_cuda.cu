#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <iostream>

#include "boundary_value_prob_cuda.cuh"
#include "utils.hpp"
#include "utils_cuda.cuh"

/* LU decomposition kernel
 * l_sub_diag[i] = sub_diag[i] / u_main_diag[i - 1];
 * u_main_diag[i] = main_diag[i] - l_sub_diag[i] * super_diag[i - 1];
 */
template <typename T>
__global__ void lu_decomposition_kernel(const int n, T *sub_diag, T *main_diag,
                                        T *super_diag, T *l_sub_diag,
                                        T *u_main_diag, Precision prec) {
  /* initialize */
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /* compute the LU decomposition */
  u_main_diag[0] = main_diag[0];
  for (int i = 1; i < n; i++) {
    if (prec == Double) {
      l_sub_diag[i] = __ddiv_rn(sub_diag[i], u_main_diag[i - 1]);
      u_main_diag[i] =
          __dsub_rn(main_diag[i], __dmul_rn(l_sub_diag[i], super_diag[i - 1]));
    } else if (prec == Single) {
      l_sub_diag[i] = __fdiv_rn(sub_diag[i], u_main_diag[i - 1]);
      u_main_diag[i] =
          __fsub_rn(main_diag[i], __fmul_rn(l_sub_diag[i], super_diag[i - 1]));
    } else if (prec == Half) {
      l_sub_diag[i] = __hdiv(sub_diag[i], u_main_diag[i - 1]);
      u_main_diag[i] =
          __hsub_rn(main_diag[i], __hmul_rn(l_sub_diag[i], super_diag[i - 1]));
    } else {
      printf("<Cuda Error>: Invalid precision\n");
      return;
    }
  }
}

/* forward substituion kernel
  forward_sol[i] = rhs[i] - l_sub_diag[i] * forward_sol[i - 1];
*/
template <typename T>
__global__ void forward_substitution_kernel(const int n, T *l_sub_diag, T *rhs,
                                            T *forward_sol, Precision prec) {
  /* initialize */
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /* forward substitution */
  forward_sol[0] = rhs[0];

  for (int i = 1; i < n; i++) {
    if (prec == Double) {
      forward_sol[i] =
          __dsub_rn(rhs[i], __dmul_rn(l_sub_diag[i], forward_sol[i - 1]));
    } else if (prec == Single) {
      forward_sol[i] =
          __fsub_rn(rhs[i], __fmul_rn(l_sub_diag[i], forward_sol[i - 1]));
    } else if (prec == Half) {
      forward_sol[i] =
          __hsub_rn(rhs[i], __hmul_rn(l_sub_diag[i], forward_sol[i - 1]));
    } else {
      printf("<Cuda Error>: Invalid precision\n");
      return;
    }
  }
}

/* backward substituion kernel
  state[i] = (forward_sol[i] - super_diag[i] * state[i + 1]) / u_main_diag[i];
 */
template <typename T>
__global__ void backward_substitution_kernel(const int n, T *u_main_diag,
                                             T *forward_sol, T *super_diag,
                                             T *state, Precision prec) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  /* backward substituion */
  if (prec == Double) {
    state[n - 1] = __ddiv_rn(forward_sol[n - 1], u_main_diag[n - 1]);
  } else if (prec == Single) {
    state[n - 1] = __fdiv_rn(forward_sol[n - 1], u_main_diag[n - 1]);
  } else if (prec == Half) {
    state[n - 1] = __hdiv(forward_sol[n - 1], u_main_diag[n - 1]);
  } else {
    printf("<Cuda Error>: Invalid precision\n");
    return;
  }

  for (int i = n - 2; i > -1; i--) {
    if (prec == Double) {
      state[i] = __ddiv_rn(
          __dsub_rn(forward_sol[i], __dmul_rn(super_diag[i], state[i + 1])),
          u_main_diag[i]);
    } else if (prec == Single) {
      state[i] = __fdiv_rn(
          __fsub_rn(forward_sol[i], __fmul_rn(super_diag[i], state[i + 1])),
          u_main_diag[i]);
    } else if (prec == Half) {
      state[i] = __hdiv(
          __hsub_rn(forward_sol[i], __hmul_rn(super_diag[i], state[i + 1])),
          u_main_diag[i]);
    }
  }
}

/* state integral kernel */
template <typename T>
__global__ void state_integral_kernel(const int n, T *state, T *state_integral,
                                      Precision prec) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  T delta_x = static_cast<T>(1.0) / (n + 1);
  T sum = static_cast<T>(0.0);
  for (int i = 0; i < n; i++) {
    if (prec == Double) {
      sum = __dadd_rn(sum, state[i]);
    } else if (prec == Single) {
      sum = __fadd_rn(sum, state[i]);
    } else if (prec == Half) {
      sum = __hadd_rn(sum, state[i]);
    }
  }

  if (prec == Double) {
    *state_integral = __dmul_rn(delta_x, sum);
  } else if (prec == Single) {
    *state_integral = __fmul_rn(delta_x, sum);
  } else if (prec == Half) {
    *state_integral = __hmul_rn(delta_x, sum);
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

  lu_decomposition_kernel<<<gridDim, blockDim>>>(Ns, d_sub_diag, d_main_diag,
                                                 d_super_diag, d_l_sub_diag,
                                                 d_u_main_diag, prec);
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

/* forward substituion kernel launcher */
template <typename T>
void launch_forward_substitution_kernel(const int num_intervals,
                                        const std::vector<T> &h_l_sub_diag,
                                        const std::vector<T> &h_rhs,
                                        std::vector<T> &h_forward_sol,
                                        Precision prec, bool verbose = false) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  int size = Ns * sizeof(T);
  T *d_l_sub_diag, *d_rhs, *d_forward_sol;
  /* allocate memory */
  cudaCheck(cudaMalloc((void **)&d_l_sub_diag, size));
  cudaCheck(cudaMalloc((void **)&d_rhs, size));
  cudaCheck(cudaMalloc((void **)&d_forward_sol, size));
  /* host to device */
  cudaCheck(cudaMemcpy(d_l_sub_diag, h_l_sub_diag.data(), size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_rhs, h_rhs.data(), size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for forward substitution in "
              << to_string(prec) << " precision" << std::endl;
  forward_substitution_kernel<<<gridDim, blockDim>>>(Ns, d_l_sub_diag, d_rhs,
                                                     d_forward_sol, prec);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(cudaMemcpy(h_forward_sol.data(), d_forward_sol, size,
                       cudaMemcpyDeviceToHost));
  /* free */
  cudaCheck(cudaFree(d_l_sub_diag));
  cudaCheck(cudaFree(d_rhs));
  cudaCheck(cudaFree(d_forward_sol));
}

/* backward substituion kernel launcher */
template <typename T>
void launch_backward_substitution_kernel(const int num_intervals,
                                         const std::vector<T> h_u_main_diag,
                                         const std::vector<T> h_forward_sol,
                                         const std::vector<T> h_super_diag,
                                         std::vector<T> &h_state,
                                         Precision prec, bool verbose = false) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  int size = Ns * sizeof(T);
  T *d_u_main_diag, *d_forward_sol, *d_super_diag, *d_state;
  /* allocate */
  cudaCheck(cudaMalloc((void **)&d_u_main_diag, size));
  cudaCheck(cudaMalloc((void **)&d_forward_sol, size));
  cudaCheck(cudaMalloc((void **)&d_super_diag, size));
  cudaCheck(cudaMalloc((void **)&d_state, size));
  /* host to device */
  cudaCheck(cudaMemcpy(d_u_main_diag, h_u_main_diag.data(), size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_forward_sol, h_forward_sol.data(), size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_super_diag, h_super_diag.data(), size,
                       cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for backward substitution in "
              << to_string(prec) << " precision" << std::endl;
  backward_substitution_kernel<<<gridDim, blockDim>>>(
      Ns, d_u_main_diag, d_forward_sol, d_super_diag, d_state, prec);
  cudaCheck(cudaGetLastError());
  /* device to host */
  cudaCheck(cudaMemcpy(h_state.data(), d_state, size, cudaMemcpyDeviceToHost));
  /* free */
  cudaCheck(cudaFree(d_u_main_diag));
  cudaCheck(cudaFree(d_forward_sol));
  cudaCheck(cudaFree(d_super_diag));
  cudaCheck(cudaFree(d_state));
}

/* state integral kernel launcher */
template <typename T>
void launch_state_integral_kernel(const int num_intervals,
                                  std::vector<T> &h_state, T &h_state_integral,
                                  Precision prec, bool verbose = false) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  const int size = Ns * sizeof(T);
  T *d_state;
  T *d_state_integral;
  /* allocate */
  cudaCheck(cudaMalloc((void **)&d_state, size));
  cudaCheck(cudaMalloc((void **)&d_state_integral, sizeof(T)));
  /* host to device */
  cudaCheck(cudaMemcpy(d_state, h_state.data(), size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for integrating ode solution in "
              << to_string(prec) << " precision" << std::endl;
  state_integral_kernel<<<gridDim, blockDim>>>(Ns, d_state, d_state_integral,
                                               prec);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  /* device to host */
  cudaCheck(cudaMemcpy(&h_state_integral, d_state_integral, sizeof(T),
                       cudaMemcpyDeviceToHost));
  /* free */
  cudaCheck(cudaFree(d_state));
  cudaCheck(cudaFree(d_state_integral));
}

/* launch thomas algorithm kernel(s) */
template <typename T>
void launch_thomas_algorithm_kernel(const int num_intervals,
                                    const std::vector<T> &h_sub_diag,
                                    const std::vector<T> &h_main_diag,
                                    const std::vector<T> &h_super_diag,
                                    const std::vector<T> &h_rhs,
                                    std::vector<T> &h_state, Precision prec,
                                    bool verbose) {
  // initialize
  const int Ns = num_intervals - 1;
  std::vector<T> h_l_sub_diag(Ns), h_u_main_diag(Ns), h_forward_sol(Ns);

  // LU decomposition
  launch_lu_decomposition_kernel<T>(num_intervals, h_sub_diag, h_main_diag,
                                    h_super_diag, h_l_sub_diag, h_u_main_diag,
                                    prec);
  /* forward substitution */
  launch_forward_substitution_kernel<T>(num_intervals, h_l_sub_diag, h_rhs,
                                        h_forward_sol, prec);
  /* backward substitution */
  launch_backward_substitution_kernel<T>(
      num_intervals, h_u_main_diag, h_forward_sol, h_super_diag, h_state, prec);

  if (verbose == true) {
    printf("Thomas algorithm computed state in %s precision\n",
           to_string(prec).c_str());
    for (auto &a : h_state) {
      printf("%f\n", a);
    }
  }
}

/* launch ode state integral kernerl(s) */
template <typename T>
void launch_ode_state_integral_kernel(const int num_intervals,
                                      const std::vector<T> &h_sub_diag,
                                      const std::vector<T> &h_main_diag,
                                      const std::vector<T> &h_super_diag,
                                      const std::vector<T> &h_rhs,
                                      T &h_state_integral, Precision prec,
                                      bool verbose) {
  /* initialize */
  const int Ns = num_intervals - 1;
  std::vector<T> h_state(Ns);
  /* launch thomas algorithm */
  launch_thomas_algorithm_kernel<T>(num_intervals, h_sub_diag, h_main_diag,
                                    h_super_diag, h_rhs, h_state, prec,
                                    verbose);
  /* launch state integral kernel */
  launch_state_integral_kernel<T>(num_intervals, h_state, h_state_integral,
                                  prec);
  /* print */
  if (verbose == true) {
    printf("State integral: %f\n", h_state_integral);
  }
}

/* template initialization */
template void launch_thomas_algorithm_kernel<double>(
    const int, const std::vector<double> &, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    std::vector<double> &, Precision, bool);

template void launch_thomas_algorithm_kernel<float>(
    const int, const std::vector<float> &, const std::vector<float> &,
    const std::vector<float> &, const std::vector<float> &,
    std::vector<float> &, Precision, bool);

template void launch_thomas_algorithm_kernel<half>(
    const int, const std::vector<half> &, const std::vector<half> &,
    const std::vector<half> &, const std::vector<half> &, std::vector<half> &,
    Precision, bool);

template void launch_ode_state_integral_kernel<double>(
    const int, const std::vector<double> &, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &, double &,
    Precision, bool);
