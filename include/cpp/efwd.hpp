#ifndef EFWD_HPP
#define EFWD_HPP

#include "definition.hpp"

template <typename T>
void computeForwardErrorThomas(int N, double *sub_diag, double *main_diag, double *super_diag, double *rhs, T *a, T *b, T *u, double *ebwd, double *efwd, double *C);

template <typename T>
void computeForwardErrorQoi(int N, T *u, double *C, double *efwd_bound, BoundType btype, Precision prec, double confidence = 0.99);

#endif
