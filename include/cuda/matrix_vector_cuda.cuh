#ifndef MATRIX_VECTOR_CUDA_CUH
#define MATRIX_VECTOR_CUDA_CUH

#include <vector>

#include "definition.hpp"
#include "gamma.hpp"
#include "utils.hpp"

/* matrix-vector product kernel */
template <typename T>
void launch_matvec_product_kernel(const Matrix<T> &h_matrix,
                                  const std::vector<T> &h_a,
                                  std::vector<T> &h_result, Precision prec,
                                  bool verbose = false);

#endif
