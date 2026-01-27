#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <iostream>

#include "boundary_value_prob_cuda.cuh"
#include "rounding_error_model.cuh"
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

/* LU decomposition model kernel
 * l_sub_diag[i] = sub_diag[i] / u_main_diag[i - 1];
 * u_main_diag[i] = main_diag[i] - l_sub_diag[i] * super_diag[i - 1];
 */
template <BVPKernelTag K>
__global__ void lu_decomposition_model_kernel(
    const int n, double *sub_diag, double *main_diag, double *super_diag,
    double *l_sub_diag, double *u_main_diag, Precision prec,
    BoundModel bound_model, const double beta_dist_alpha,
    const double beta_dist_beta, const int experiment_id,
    unsigned long long seed = 1234ULL) {
  /* initialize */
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int threads = gridDim.x * blockDim.x;
  constexpr int kernel_id = static_cast<int>(K);
  /* roundig error */
  const int num_perturb = 3;
  double rounding_error[num_perturb];
  /* random state */
  curandState state;
  long long sequence =
      (long long)experiment_id * (int)BVPKernelTag::Count * threads +
      kernel_id * threads + gid;
  curand_init(seed, sequence, 0, &state);

  /* compute the LU decomposition */
  u_main_diag[0] = main_diag[0];
  for (int i = 1; i < n; i++) {
    /* sample rounding error delta */
    sample_rounding_error_distribution(num_perturb, rounding_error, prec,
                                       bound_model, beta_dist_alpha,
                                       beta_dist_beta, &state);
    for (int j = 0; j < num_perturb; j++)
      rounding_error[j] = 1.0 + rounding_error[j];

    /* compute */
    l_sub_diag[i] = __ddiv_rn(sub_diag[i],
                              __dmul_rn(u_main_diag[i - 1], rounding_error[0]));

    u_main_diag[i] = __ddiv_rn(
        __dsub_rn(main_diag[i],
                  __dmul_rn(l_sub_diag[i],
                            __dmul_rn(super_diag[i - 1], rounding_error[1]))),
        rounding_error[2]);
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

/* forward substituion model kernel
  forward_sol[i] = rhs[i] - l_sub_diag[i] * forward_sol[i - 1];
*/
template <BVPKernelTag K>
__global__ void forward_substitution_model_kernel(
    const int n, double *l_sub_diag, double *rhs, double *forward_sol,
    Precision prec, BoundModel bound_model, const double beta_dist_alpha,
    const double beta_dist_beta, const int experiment_id,
    unsigned long long seed = 1234ULL) {
  /* initialize */
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int threads = gridDim.x * blockDim.x;
  constexpr int kernel_id = static_cast<int>(K);
  /* roundig error */
  const int num_perturb = 2;
  double rounding_error[num_perturb];
  /* random state */
  curandState state;
  long long sequence =
      (long long)experiment_id * (int)BVPKernelTag::Count * threads +
      kernel_id * threads + gid;
  curand_init(seed, sequence, 0, &state);

  /* forward substitution */
  forward_sol[0] = rhs[0];

  for (int i = 1; i < n; i++) {
    /* sample rounding error delta */
    sample_rounding_error_distribution(num_perturb, rounding_error, prec,
                                       bound_model, beta_dist_alpha,
                                       beta_dist_beta, &state);
    for (int j = 0; j < num_perturb; j++) {
      rounding_error[j] = 1.0 + rounding_error[j];
    }

    forward_sol[i] = __ddiv_rn(
        __dsub_rn(rhs[i],
                  __dmul_rn(l_sub_diag[i],
                            __dmul_rn(forward_sol[i - 1], rounding_error[0]))),
        rounding_error[1]);
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

/* backward substituion model kernel
  state[i] = (forward_sol[i] - super_diag[i] * state[i + 1]) / u_main_diag[i];
 */
template <BVPKernelTag K>
__global__ void backward_substitution_model_kernel(
    const int n, double *u_main_diag, double *forward_sol, double *super_diag,
    double *state, Precision prec, BoundModel bound_model,
    const double beta_dist_alpha, const double beta_dist_beta,
    const int experiment_id, unsigned long long seed = 1234ULL) {
  // initialize
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int threads = gridDim.x * blockDim.x;
  constexpr int kernel_id = static_cast<int>(K);
  /* roundig error */
  const int num_perturb = 4;
  double rounding_error[num_perturb];
  /* random state */
  curandState randstate;
  long long sequence =
      (long long)experiment_id * (int)BVPKernelTag::Count * threads +
      kernel_id * threads + gid;
  curand_init(seed, sequence, 0, &randstate);

  /* backward substituion */
  sample_rounding_error_distribution(num_perturb, rounding_error, prec,
                                     bound_model, beta_dist_alpha,
                                     beta_dist_beta, &randstate);
  for (int j = 0; j < num_perturb; j++) {
    rounding_error[j] = 1.0 + rounding_error[j];
  }

  state[n - 1] = __ddiv_rn(forward_sol[n - 1],
                           __dmul_rn(u_main_diag[n - 1], rounding_error[0]));

  for (int i = n - 2; i > -1; i--) {
    // sample rounding error
    sample_rounding_error_distribution(num_perturb, rounding_error, prec,
                                       bound_model, beta_dist_alpha,
                                       beta_dist_beta, &randstate);
    for (int j = 0; j < num_perturb; j++) {
      rounding_error[j] = 1.0 + rounding_error[j];
    }

    // solve
    state[i] = __ddiv_rn(
        __dsub_rn(forward_sol[i],
                  __dmul_rn(super_diag[i],
                            __dmul_rn(state[i + 1], rounding_error[1]))),
        __dmul_rn(u_main_diag[i],
                  __dmul_rn(rounding_error[2], rounding_error[3])));
  }
}

/* state integral kernel */
template <typename T>
__global__ void state_integral_kernel(const int n, T *state, T *state_integral,
                                      Precision prec) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  T delta_x = static_cast<T>(1.0 / (n + 1));
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

/* state integral model kernel */
template <BVPKernelTag K>
__global__ void state_integral_model_kernel(
    const int n, double *state, double *state_integral, Precision prec,
    BoundModel bound_model, const double beta_dist_alpha,
    const double beta_dist_beta, const int experiment_id,
    unsigned long long seed = 1234ULL) {
  // initialize
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int threads = gridDim.x * blockDim.x;
  constexpr int kernel_id = static_cast<int>(K);
  /* roundig error */
  const int num_perturb = 1;
  double rounding_error[num_perturb];
  /* random state */
  curandState randstate;
  long long sequence =
      (long long)experiment_id * (int)BVPKernelTag::Count * threads +
      kernel_id * threads + gid;
  curand_init(seed, sequence, 0, &randstate);

  double delta_x = static_cast<double>(1.0 / (n + 1));
  double sum = static_cast<double>(0.0);
  for (int i = 0; i < n; i++) {
    // sample rounding error
    sample_rounding_error_distribution(num_perturb, rounding_error, prec,
                                       bound_model, beta_dist_alpha,
                                       beta_dist_beta, &randstate);
    for (int j = 0; j < num_perturb; j++) {
      rounding_error[j] = 1.0 + rounding_error[j];
    }

    // compute
    sum = __dadd_rn(sum, __dmul_rn(state[i], rounding_error[0]));
  }

  // normalize
  sample_rounding_error_distribution(num_perturb, rounding_error, prec,
                                     bound_model, beta_dist_alpha,
                                     beta_dist_beta, &randstate);
  for (int j = 0; j < num_perturb; j++) {
    rounding_error[j] = 1.0 + rounding_error[j];
  }

  *state_integral = __dmul_rn(delta_x, __dmul_rn(sum, rounding_error[0]));
}

/* monte carlo kernel */
template <typename T>
__global__ void monte_carlo_expectation_kernel(const int num_samples,
                                               T *integrand, T *result,
                                               Precision prec) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  /* compute the summation */
  T sum = static_cast<T>(0.0);
  for (int i = 0; i < num_samples; i++) {
    if (prec == Double) {
      sum = __dadd_rn(sum, integrand[i]);
    } else if (prec == Single) {
      sum = __fadd_rn(sum, integrand[i]);
    } else if (prec == Half) {
      sum = __hadd_rn(sum, static_cast<half>(integrand[i]));
    } else {
      printf("<Cuda Error>: Invalid precision\n");
      return;
    }
  }

  /* multiply */
  if (prec == Double) {
    *result = __dmul_rn(sum, static_cast<double>(1.0 / num_samples));
  } else if (prec == Single) {
    *result = __fmul_rn(sum, static_cast<float>(1.0 / num_samples));
  } else if (prec == Half) {
    *result = __hmul_rn(sum, static_cast<half>(1.0 / num_samples));
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

/* LU decomposition model kernel launcher */
void launch_lu_decomposition_model_kernel(
    const int num_intervals, const std::vector<double> &h_sub_diag,
    const std::vector<double> &h_main_diag,
    const std::vector<double> &h_super_diag, std::vector<double> &h_l_sub_diag,
    std::vector<double> &h_u_main_diag, Precision prec,
    const gamma_config &gamma_cfg, const int experiment_id,
    bool verbose = false) {
  /* /1* kernel parameters *1/ */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  int size = Ns * sizeof(double);
  double *d_sub_diag, *d_main_diag, *d_super_diag, *d_l_sub_diag,
      *d_u_main_diag;
  BoundModel bound_model = gamma_cfg.bound_model;
  const double beta_dist_alpha = gamma_cfg.beta_dist_alpha;
  const double beta_dist_beta = gamma_cfg.beta_dist_beta;

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
  /* /1* launch kernel *1/ */
  if (verbose == true)
    std::cout << "launching kernel for LU decomposition in "
              << to_string(Double) << " precision" << std::endl;

  lu_decomposition_model_kernel<BVPKernelTag::LUDecomposition>
      <<<gridDim, blockDim>>>(Ns, d_sub_diag, d_main_diag, d_super_diag,
                              d_l_sub_diag, d_u_main_diag, prec, bound_model,
                              beta_dist_alpha, beta_dist_beta, experiment_id);
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

/* forward substituion model kernel launcher */
void launch_forward_substitution_model_kernel(
    const int num_intervals, const std::vector<double> &h_l_sub_diag,
    const std::vector<double> &h_rhs, std::vector<double> &h_forward_sol,
    Precision prec, const gamma_config &gamma_cfg, const int experiment_id,
    bool verbose = false) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  int size = Ns * sizeof(double);
  double *d_l_sub_diag, *d_rhs, *d_forward_sol;
  BoundModel bound_model = gamma_cfg.bound_model;
  const double beta_dist_alpha = gamma_cfg.beta_dist_alpha;
  const double beta_dist_beta = gamma_cfg.beta_dist_beta;

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
              << to_string(Double) << " precision" << std::endl;
  forward_substitution_model_kernel<BVPKernelTag::ForwardSubstitution>
      <<<gridDim, blockDim>>>(Ns, d_l_sub_diag, d_rhs, d_forward_sol, prec,
                              bound_model, beta_dist_alpha, beta_dist_beta,
                              experiment_id);
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

/* backward substituion model kernel launcher */
void launch_backward_substitution_model_kernel(
    const int num_intervals, const std::vector<double> h_u_main_diag,
    const std::vector<double> h_forward_sol,
    const std::vector<double> h_super_diag, std::vector<double> &h_state,
    Precision prec, const gamma_config &gamma_cfg, const int experiment_id,
    bool verbose = false) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  int size = Ns * sizeof(double);
  double *d_u_main_diag, *d_forward_sol, *d_super_diag, *d_state;
  BoundModel bound_model = gamma_cfg.bound_model;
  const double beta_dist_alpha = gamma_cfg.beta_dist_alpha;
  const double beta_dist_beta = gamma_cfg.beta_dist_beta;
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
              << to_string(Double) << " precision" << std::endl;
  backward_substitution_model_kernel<BVPKernelTag::BackwardSubstitution>
      <<<gridDim, blockDim>>>(Ns, d_u_main_diag, d_forward_sol, d_super_diag,
                              d_state, prec, bound_model, beta_dist_alpha,
                              beta_dist_beta, experiment_id);
  cudaCheck(cudaGetLastError());
  // device to host
  cudaCheck(cudaMemcpy(h_state.data(), d_state, size, cudaMemcpyDeviceToHost));
  // free
  cudaCheck(cudaFree(d_u_main_diag));
  cudaCheck(cudaFree(d_forward_sol));
  cudaCheck(cudaFree(d_super_diag));
  cudaCheck(cudaFree(d_state));
}

/* state integral kernel launcher */
template <typename T>
void launch_state_integral_kernel(const int num_intervals,
                                  std::vector<T> &h_state, T &h_state_integral,
                                  Precision prec, bool verbose) {
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

/* state integral model kernel launcher */
void launch_state_integral_model_kernel(const int num_intervals,
                                        std::vector<double> &h_state,
                                        double &h_state_integral,
                                        Precision prec,
                                        const gamma_config &gamma_cfg,
                                        const int experiment_id, bool verbose) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialize */
  const int Ns = num_intervals - 1;
  const int size = Ns * sizeof(double);
  double *d_state;
  double *d_state_integral;
  BoundModel bound_model = gamma_cfg.bound_model;
  const double beta_dist_alpha = gamma_cfg.beta_dist_alpha;
  const double beta_dist_beta = gamma_cfg.beta_dist_beta;

  /* allocate */
  cudaCheck(cudaMalloc((void **)&d_state, size));
  cudaCheck(cudaMalloc((void **)&d_state_integral, sizeof(double)));
  /* host to device */
  cudaCheck(cudaMemcpy(d_state, h_state.data(), size, cudaMemcpyHostToDevice));
  /* launch kernel */
  if (verbose == true)
    std::cout << "launching kernel for integrating ode solution in "
              << to_string(Double) << " precision" << std::endl;
  state_integral_model_kernel<BVPKernelTag::StateIntegral>
      <<<gridDim, blockDim>>>(Ns, d_state, d_state_integral, prec, bound_model,
                              beta_dist_alpha, beta_dist_beta, experiment_id);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  /* device to host */
  cudaCheck(cudaMemcpy(&h_state_integral, d_state_integral, sizeof(double),
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
      printf("%f\n", static_cast<double>(a));
    }
  }
}

/* launch thomas algorithm model kernel */
void launch_thomas_algorithm_model_kernel(
    const int num_intervals, const std::vector<double> &h_sub_diag,
    const std::vector<double> &h_main_diag,
    const std::vector<double> &h_super_diag, const std::vector<double> &h_rhs,
    std::vector<double> &h_state, Precision prec, const gamma_config &gamma_cfg,
    const int experiment_id, bool verbose) {
  // initialize
  const int Ns = num_intervals - 1;
  std::vector<double> h_l_sub_diag(Ns), h_u_main_diag(Ns), h_forward_sol(Ns);

  // LU decomposition
  launch_lu_decomposition_model_kernel(
      num_intervals, h_sub_diag, h_main_diag, h_super_diag, h_l_sub_diag,
      h_u_main_diag, prec, gamma_cfg, experiment_id);

  /* forward substitution */
  launch_forward_substitution_model_kernel(num_intervals, h_l_sub_diag, h_rhs,
                                           h_forward_sol, prec, gamma_cfg,
                                           experiment_id);

  /* backward substitution */
  launch_backward_substitution_model_kernel(
      num_intervals, h_u_main_diag, h_forward_sol, h_super_diag, h_state, prec,
      gamma_cfg, experiment_id);

  for (int i = 0; i < Ns; i++) {
    printf("%f %f %f %f\n", h_l_sub_diag[i], h_u_main_diag[i], h_forward_sol[i],
           h_state[i]);
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
    printf("State integral: %f\n", static_cast<double>(h_state_integral));
  }
}

/* launch ode state integral model kernerl(s) */
void launch_ode_state_integral_model_kernel(
    const int num_intervals, const std::vector<double> &h_sub_diag,
    const std::vector<double> &h_main_diag,
    const std::vector<double> &h_super_diag, const std::vector<double> &h_rhs,
    double &h_state_integral, Precision prec, const gamma_config &gamma_cfg,
    const int experiment_id, bool verbose) {
  /* initialize */
  const int Ns = num_intervals - 1;
  std::vector<double> h_state(Ns);
  /* launch thomas algorithm */
  launch_thomas_algorithm_model_kernel(num_intervals, h_sub_diag, h_main_diag,
                                       h_super_diag, h_rhs, h_state, prec,
                                       gamma_cfg, experiment_id, verbose);
  /* launch state integral kernel */
  launch_state_integral_model_kernel(num_intervals, h_state, h_state_integral,
                                     prec, gamma_cfg, experiment_id);
  /* print */
  if (verbose == true) {
    printf("State integral: %f\n", static_cast<double>(h_state_integral));
  }
}

/* launch |\hat{L}|{\hat{U}| kernel
  Kernel first computes the LU decomposition in the prescribed precision and
  then casts it to double precison to compute the |\hat{L}||\hat{U}|, where
  \hat{L} and \hat{U} are the computed factors. This is required for the
  backward error analysis for solving a linear system. Notes:
  1. See eq. 4.4 in A NEW APPROACH TO PROBABILISTIC ROUNDING ERROR ANALYSIS.
  2. output h_abs_lu_mult_true is stored in row-major
*/
template <typename T>
void launch_abs_lu_multiplication_kernel(
    const int num_intervals, const std::vector<T> &h_sub_diag,
    const std::vector<T> &h_main_diag, const std::vector<T> &h_super_diag,
    const std::vector<T> &h_rhs, std::vector<double> &h_abs_lu_mult_true,
    Precision prec, bool verbose) {
  // initialize
  const int Ns = num_intervals - 1;
  std::vector<T> h_l_sub_diag(Ns), h_u_main_diag(Ns);
  std::vector<double> h_l_sub_diag_abs, h_u_main_diag_abs, h_super_diag_abs;

  // LU decomposition
  launch_lu_decomposition_kernel<T>(num_intervals, h_sub_diag, h_main_diag,
                                    h_super_diag, h_l_sub_diag, h_u_main_diag,
                                    prec);
  // compute the absolute value
  convert_vector_to_absolute_double(h_l_sub_diag, h_l_sub_diag_abs);
  convert_vector_to_absolute_double(h_u_main_diag, h_u_main_diag_abs);
  convert_vector_to_absolute_double(h_super_diag, h_super_diag_abs);

  // compute |\hat{L}||\hat{U}|
  h_abs_lu_mult_true.assign(Ns * Ns, 0.0);
  for (int i = 0; i < Ns; i++) {
    // main diagonal
    double main_val = h_u_main_diag_abs[i];
    if (i > 0) {
      main_val += h_l_sub_diag_abs[i] * h_super_diag_abs[i - 1];
    }
    h_abs_lu_mult_true[i * Ns + i] = main_val;

    // super diagonal
    if (i < Ns - 1) {
      h_abs_lu_mult_true[i * Ns + (i + 1)] = h_super_diag_abs[i];
    }

    // sub diagonal
    if (i > 0) {
      h_abs_lu_mult_true[i * Ns + (i - 1)] =
          h_l_sub_diag_abs[i] * h_u_main_diag_abs[i - 1];
    }
  }
}

template <typename T>
void launch_monte_carlo_expectation_kernel(const std::vector<T> &h_integrand,
                                           T &h_result, Precision prec,
                                           bool verbose) {
  /* kernel parameters */
  dim3 blockDim = 1;
  dim3 gridDim = 1;
  /* initialization */
  const int num_samples = h_integrand.size();
  T *d_integrand, *d_result;
  const int size = num_samples * sizeof(T);
  /* memory allocation */
  cudaCheck(cudaMalloc((void **)&d_integrand, size));
  cudaCheck(cudaMalloc((void **)&d_result, sizeof(T)));
  /* host to device */
  cudaCheck(cudaMemcpy(d_integrand, h_integrand.data(), size,
                       cudaMemcpyHostToDevice));

  if (verbose == true)
    std::cout << "launching Monte-Carlo integral kernel using " << num_samples
              << "for in" << to_string(prec) << " precision" << std::endl;

  monte_carlo_expectation_kernel<<<gridDim, blockDim>>>(
      num_samples, d_integrand, d_result, prec);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());

  /* device to host */
  cudaCheck(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

  /* free */
  cudaCheck(cudaFree(d_integrand));
  cudaCheck(cudaFree(d_result));
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

template void launch_ode_state_integral_kernel<float>(
    const int, const std::vector<float> &, const std::vector<float> &,
    const std::vector<float> &, const std::vector<float> &, float &, Precision,
    bool);

template void launch_ode_state_integral_kernel<half>(const int,
                                                     const std::vector<half> &,
                                                     const std::vector<half> &,
                                                     const std::vector<half> &,
                                                     const std::vector<half> &,
                                                     half &, Precision, bool);

template void launch_abs_lu_multiplication_kernel(
    const int, const std::vector<double> &, const std::vector<double> &,
    const std::vector<double> &, const std::vector<double> &,
    std::vector<double> &, Precision, bool);

template void launch_abs_lu_multiplication_kernel(
    const int, const std::vector<float> &, const std::vector<float> &,
    const std::vector<float> &, const std::vector<float> &,
    std::vector<double> &, Precision, bool);

template void launch_abs_lu_multiplication_kernel(
    const int, const std::vector<half> &, const std::vector<half> &,
    const std::vector<half> &, const std::vector<half> &, std::vector<double> &,
    Precision, bool);

template void launch_state_integral_kernel<double>(const int,
                                                   std::vector<double> &,
                                                   double &, Precision, bool);
template void launch_state_integral_kernel<float>(const int,
                                                  std::vector<float> &, float &,
                                                  Precision, bool);
template void launch_state_integral_kernel<half>(const int, std::vector<half> &,
                                                 half &, Precision, bool);

template void launch_monte_carlo_expectation_kernel<double>(
    const std::vector<double> &h_integrad, double &h_integral, Precision prec,
    bool verbose);
template void launch_monte_carlo_expectation_kernel<float>(
    const std::vector<float> &h_integrad, float &h_integral, Precision prec,
    bool verbose);
template void launch_monte_carlo_expectation_kernel<half>(
    const std::vector<half> &h_integrad, half &h_integral, Precision prec,
    bool verbose);
