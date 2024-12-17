#ifndef EFWD_HPP
#define EFWD_HPP

template <typename T>
void computeForwardErrorThomas(int N, double *sub_diag, double *main_diag, double *super_diag, double *rhs, T *a, T *b, T *u, double *ebwd, double *efwd);

#endif
