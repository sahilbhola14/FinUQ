#ifndef DOT_PRODUCT_CUDA_CUH
#define DOT_PRODUCT_CUDA_CUH

#include <vector>

#include "definition.hpp"

/* dot product kernel */
template <typename T>
void launch_sequential_dot_product_kernel(const int n,
                                          const std::vector<T> &h_a,
                                          const std::vector<T> &h_b,
                                          T *h_result, Precision prec,
                                          bool verbose = false);
void launch_sequential_dot_product_model_kernel(
    const int n, const std::vector<double> &h_a, const std::vector<double> &h_b,
    double *h_result, Precision prec, BoundModel bound_model,
    bool verbose = false);

#endif
