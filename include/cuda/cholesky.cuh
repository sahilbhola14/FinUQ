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

// forward solve: Ly = b
template <typename T>
void launch_forward_solve_kernel(const int N, const std::vector<T> &h_l,
                                 const std::vector<T> &h_b, std::vector<T> &h_y,
                                 Precision prec);

// backward solve: L^T x = y
template <typename T>
void launch_backward_solve_kernel(const int N, const std::vector<T> &h_l,
                                  const std::vector<T> &h_y,
                                  std::vector<T> &h_x, Precision prec);

// cholesky solve: LL^T x = rhs
template <typename T>
void launch_cholesky_solve_kernel(const int N, const std::vector<T> &h_l,
                                  const std::vector<T> &h_rhs,
                                  std::vector<T> &h_result, Precision prec);

// compute cholesly decompositon: A = LL^T, where A is of size N * N
template <typename T>
void launch_cholesky_decomposition_kernel(const int N, std::vector<T> &h_a,
                                          std::vector<T> &h_l, Precision prec);
#endif
