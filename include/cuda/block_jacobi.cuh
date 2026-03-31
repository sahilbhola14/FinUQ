#ifndef BLOCK_JACOBI_CUH
#define BLOCK_JACOBI_CUH

#include <vector>

#include "definition.hpp"
#include "poisson.hpp"

template <typename T>
void launch_block_jacobi_solver(const poisson_rhs_config<T> &poisson_rhs_cfg,
                                std::vector<T> &h_coeff,
                                std::vector<T> &h_state_initial,
                                const poisson_config &poisson_cfg,
                                bool verbose = false);

#endif
