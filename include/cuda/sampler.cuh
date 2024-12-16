#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "definition.hpp"

template <typename T>
void initializeVector(int N, T *x, Distribution dtype, unsigned long long seed);

void getODEParameters(const int N, half *parameters, unsigned long long seed);

__device__ double sampleRelativeErrorKernel(double urd, curandState *state);

#endif
