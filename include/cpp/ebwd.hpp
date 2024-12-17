#ifndef EBWD_HPP
#define EBWD_HPP

template <typename T>
void computeBackwardErrorDotProduct(double *true_result, T *approx_result, double *abs_result, double *ebwd);

template <typename T>
void computeBackwardErrorMatVecMult(int N, double *true_result, T *approx_result, double *abs_result, double *ebwd);

template <typename T>
void computeBackwardErrorThomas(int N, double *sub_diag, double *main_diag, double *super_diag, double *rhs, T *a, T *b, T *u, double *ebwd);

template <typename T>
void computeAbsoluteLUTimesSol(int N, T *a, T *b, double *super_diag, T *u, double *abs_prod);

#endif
