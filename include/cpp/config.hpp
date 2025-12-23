#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "definition.hpp"

namespace config{
const int K = 0; // Sampling interval when PowTwo distribution is used
const int N = 1 << 25; // Problem size
const int blockSize = 256; // Block size for CUDA kernel
const Distribution distType = ZeroOne;
}

#endif
