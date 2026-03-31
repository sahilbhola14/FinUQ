#ifndef CHOLESKY_CUH
#define CHOLESKY_CUH

#include <vector>

#include "definition.hpp"

// compute cholesly decompositon: A = LL^T, where A is of size N * N
template <typename T>
void launch_cholesky_decomposition_kernel(const int N, std::vector<T> &h_a,
                                          std::vector<T> &h_l, Precision prec);
#endif
