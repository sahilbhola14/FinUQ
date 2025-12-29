#ifndef UTILS_CUDA_CUH
#define UTILS_CUDA_CUH

#include <cuda.h>

#include <iostream>

/* grid size */
int get_grid_size(const int n, const int blockDim);

/* inline error check */
#define cudaCheck(ans) gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("<CUDA Error>: %s in %s at line %d\n", cudaGetErrorString(err), file,
           line);
    exit(EXIT_FAILURE);
  }
}

#endif
