#ifndef CONVERT_CUH
#define CONVERT_CUH

#include <cuda_fp16.h>

// Kernels
__global__ void DoubleToFloatKernel(int N, double *x_double, float *x_float);
__global__ void DoubleToHalfKernel(int N, double *x_double, half *x_half);

__global__ void FloatToDoubleKernel(int N, float *x_float, double *x_double);
__global__ void FloatToHalfKernel(int N, float *x, half *x_half);

__global__ void HalfToDoubleKernel(int N, half *x_half, double *x_double);
__global__ void HalfToFloatKernel(int N, half *x_half, float *x_float);
#endif
