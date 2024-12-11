#ifndef CONVERT_CUH
#define CONVERT_CUH

#include <cuda_fp16.h>

void convertHalfToDouble(int N, half *x_half, double *x_double);
void convertHalfToFloat(int N, half *x_half, float *x_float);

#endif
