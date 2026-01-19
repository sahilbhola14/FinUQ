#ifndef BOUNDARY_VALUE_PROB_CUDA_CUH
#define BOUNDARY_VALUE_PROB_CUDA_CUH

#include <vector>

#include "definition.hpp"

/* thomas algorithm kernel */
template <typename T>
void launch_thomas_algorithm_kernel(const int num_intervals,
                                    const std::vector<T> &h_sub_diag,
                                    const std::vector<T> &h_main_diag,
                                    const std::vector<T> &h_super_diag,
                                    const std::vector<T> &h_rhs,
                                    std::vector<T> &h_state, Precision prec);

/* ode state integral kernel */
template <typename T>
void launch_ode_state_integral_kernel(const int num_intervals,
                                      const std::vector<T> &h_sub_diag,
                                      const std::vector<T> &h_main_diag,
                                      const std::vector<T> &h_super_diag,
                                      const std::vector<T> &h_rhs,
                                      T &h_state_integral, Precision prec,
                                      bool verbose = false);

#endif
