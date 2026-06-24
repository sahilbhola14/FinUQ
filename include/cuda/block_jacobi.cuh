#ifndef BLOCK_JACOBI_CUH
#define BLOCK_JACOBI_CUH

#include <vector>

#include "definition.hpp"
#include "poisson.hpp"
#include "utils.hpp"

/* block jacobi solver
 */
template <typename T>
void launch_block_jacobi_solver(const poisson_rhs_config<T> &poisson_rhs_cfg,
                                std::vector<T> &h_coeff,
                                std::vector<T> &h_state_initial,
                                const poisson_config &poisson_cfg,
                                bool verbose = false);

/* cholesky factors per tile
 */
template <typename T>
std::vector<Matrix<T>> compute_cholesky_per_jacobi_tile(
    const std::vector<T> &h_a, const poisson_config &poisson_cfg);

#endif
