#include <cuda_fp16.h>

#include <vector>

#include "dot_product_cuda.cuh"

template <typename T>
__global__ void dot_product_kernel(const int n, T *a, T *b, T result,
                                   Precision prec) {
  const int tid = threadIdx.x;
  const int gid = tid + blockIdx.x * blockDim.x;

  /* initialize the dot product result */
  result = static_cast<T>(0.0);
  /* /1* compute the dot product *1/ */
  /* for (int i = 0; i < n; i++){ */
  /*     if (prec == Double){ */
  /*         result = __dadd_rn(result, __dmul_rn(a[i], b[i])); */
  /*     } else if (prec == Single){ */
  /*         result = __fadd_rn(result, __fmul_rn(a[i], b[i])); */
  /*     } else if (prec == Half){ */
  /*         half a_half = static_cast<half>(a[i]); */
  /*         half b_half = static_cast<half>(b[i]); */
  /*         result = __hadd_rn(result, __hmul_rn(a_half, b_half)); */
  /*     } else { */
  /*         printf("<CUDA Error>: Invalid precision\n"); */
  /*     } */
  /* } */
}

template <typename T>
void launch_dot_product_kernel(const int n, std::vector<T> h_a, T *h_b,
                               Precision prec) {}
