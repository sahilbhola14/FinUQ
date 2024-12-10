#include "convert.cuh"

// Double to Float (Round to Nearest)
__global__ void DoubleToFloatKernel(int N, double *x_double, float *x_float) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    x_float[i] = __double2float_rn(x_double[i]);
  }
}

// Double to Half (Round to Nearest)
__global__ void DoubleToHalfKernel(int N, double *x_double, half *x_half) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    x_half[i] = __double2half(x_double[i]);
  }
}

// Float to Double
__global__ void FloatToDoubleKernel(int N, float *x_float, double *x_double) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    x_double[i] = static_cast<double>(x_float[i]);
  }
}

// Float to Half (Round to Nearest)
__global__ void FloatToHalfKernel(int N, float *x_float, half *x_half) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    x_half[i] = __float2half(x_float[i]);
  }
}

// Half to Double
__global__ void HalfToDoubleKernel(int N, half *x_half, double *x_double) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    x_double[i] = static_cast<double>(x_half[i]);
  }
}

// Half to Float
__global__ void HalfToFloatKernel(int N, half *x_half, float *x_float) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    x_float[i] = __half2float(x_half[i]);
  }
}
