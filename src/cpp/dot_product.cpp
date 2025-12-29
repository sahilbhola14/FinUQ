#include "dot_product.hpp"

#include <cuda.h>
#include <cuda_fp16.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "distribution.hpp"
#include "dot_product_cuda.cuh"
#include "utils_cuda.cuh"

/* perform sequential dot product num_experiments */
template <typename T>
void run_dot_product_impl(const int n, Precision prec, Distribution dist,
                          const int num_experiments = 100) {
  /* /1* initialization *1/ */
  /* std::vector<T> h_a(n), h_b(n), h_results(num_experiments); */

  /* printf("** Vector size: %d **\n", n); */
  /* /1* allocate the size to the device buffer *1/ */
  /* cudaCheck(cudaMalloc((void **)&d_a, size)); */
  /* cudaCheck(cudaMalloc((void **)&d_b, size)); */
  /* cudaCheck(cudaMalloc((void **)&d_result, sizeof(T))); */

  /* /1* compute *1/ */
  /* for (int i = 0; i < num_experiments; i++){ */
  /*     if (i % 10  == 0) printf("percent complete: %.2f\n",
   * float((i+1)*100/num_experiments)); */
  /*     /1* sample vectors *1/ */
  /*     sample_random_vector(h_a, prec, dist); */
  /*     sample_random_vector(h_b, prec, dist); */
  /*     /1* host to device *1/ */
  /*     cudaCheck(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
   */
  /*     cudaCheck(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
   */
  /*     /1* device computation *1/ */
  /*     dot_product_kernel<T><<<gridDim, blockDim>>>(n, d_a, d_b, d_result,
   * prec); */
  /*     /1* device to host *1/ */
  /*     /1* cudaCheck(cudaMemcpy(h_results[i], d_result, sizeof(T),
   * cudaMemcpyDeviceToHost)); *1/ */
  /* } */

  /* /1* cuda free *1/ */
  /* cudaFree(d_a); */
  /* cudaFree(d_b); */
  /* /1* cudaFree(d_result); *1/ */
}

/* run dot product experiment for fixed size */
void run_dot_product_experiment_fixed_size(const int n, Precision prec,
                                           Distribution dist,
                                           const int num_experiments = 100) {
  std::vector<T> h_a(n), h_b(n);  // vectors
}

void run_dot_product_experiment(Precision prec, Distribution dist,
                                const int num_experiments) {
  /* intialization */
  std::vector<int> n_values = {10,     100,     1000,     10000,
                               100000, 1000000, 10000000, 100000000};
  std::vector<dot_product_result> results;
  results.reserve(n_values.size());
  std::cout << std::string(50, '=') << std::endl;
  std::cout << "Dot product Experiment" << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  /* run experiment for fixed vector size */
  run_dot_product_experiment_fixed_size(n_values[0], prec, dist,
                                        num_experiments);
}
