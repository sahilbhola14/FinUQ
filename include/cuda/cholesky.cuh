#ifndef CHOLESKY_CUH
#define CHOLESKY_CUH

#include <vector>

#include "definition.hpp"

// vanilla cholesky
void compute_vanilla_cholesky(const int N, std::vector<double> &h_a,
                              std::vector<double> &h_l);

// compute Frobenius norm of A - L*L^T
template <typename T>
void compute_cholesky_error(const int N, std::vector<T> &h_a,
                            std::vector<T> &h_l);

// compute L * L^T and return as std::vector<double>
template <typename T>
std::vector<double> compute_llt(const int N, std::vector<T> &h_l);

// compute L * L^T and print
template <typename T>
void compute_and_print_llt(const int N, std::vector<T> &h_l);

// compute cholesly decompositon: A = LL^T, where A is of size N * N
template <typename T>
void launch_cholesky_decomposition_kernel(const int N, std::vector<T> &h_a,
                                          std::vector<T> &h_l, Precision prec);
#endif
