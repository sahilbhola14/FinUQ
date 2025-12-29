#include "utils_cuda.cuh"

/* grid size */
int get_grid_dim(const int n, const int blockDim) {
  int gridDim = (n + blockDim - 1) / blockDim;
  return gridDim;
}
