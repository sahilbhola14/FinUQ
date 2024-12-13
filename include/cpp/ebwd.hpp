#ifndef EBWD_HPP
#define EBWD_HPP

template <typename T>
void computeBackwardErrorDotProduct(double *true_result, T *approx_result, double *abs_result, double *ebwd);

template <typename T>
void computeBackwardErrorMatVecMult(int N, double *true_result, T *approx_result, double *abs_result, double *ebwd);

#endif
