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

/* void launch_sequential_dot_product_model_kernel( */
/*     const int n, const std::vector<double> &h_a, const std::vector<double>
 * &h_b, */
/*     double *h_result, Precision prec, const gamma_config &gamma_cfg, */
/*     const int experiment_id, bool verbose = false); */

#endif
