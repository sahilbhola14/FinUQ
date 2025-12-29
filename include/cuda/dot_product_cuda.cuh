#ifndef DOT_PRODUCT_CUDA_CUH
#define DOT_PRODUCT_CUDA_CUH

#include "definition.hpp"

/* dot product kernel */
template <typename T>
__global__ void dot_product_kernel(const int n, T *a, T *b, T result,
                                   Precision prec);

#endif
