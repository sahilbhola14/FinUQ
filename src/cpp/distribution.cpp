#include "distribution.hpp"

#include <cuda_fp16.h>

#include <iostream>
#include <random>
#include <stdexcept>

/* sample from N(0,1) */
template <typename T>
void sample_normal_vector(std::vector<T> &vector, Precision prec,
                          const int seed = 42) {
  /* seed */
  std::mt19937 gen(seed);
  /* normal distribution (double)*/
  std::normal_distribution<double> normal(0.0, 1.0);
  /* sample random vector */
  for (auto &x : vector) {
    x = static_cast<T>(normal(gen));
  }
}

/* sample from U(lower, upper) */
template <typename T>
void sample_uniform_distribution(std::vector<T> &vector, const double lower,
                                 const double upper, Precision prec,
                                 const int seed = 42) {
  /* seed */
  std::mt19937 gen(seed);
  /* uniform distribution (double)*/
  std::uniform_real_distribution<double> uniform(lower, upper);
  /* sample random vector */
  for (auto &x : vector) {
    x = static_cast<T>(uniform(gen));
  }
}

/* constnat initialization */
template <typename T>
void constant_initialization(std::vector<T> &vector, const double constant,
                             Precision prec) {
  /* initialize */
  for (auto &x : vector) {
    x = static_cast<T>(constant);
  }
}

/* sample random vector */
template <typename T>
void sample_random_vector(std::vector<T> &vector, Precision prec,
                          Distribution dist, const int pow2k, const int seed) {
  if (dist == Normal) {
    sample_normal_vector(vector, prec, seed);
  } else if (dist == ZeroOne) {
    sample_uniform_distribution(vector, 0.0, 1.0, prec, seed);
  } else if (dist == MinusOnePlusOne) {
    sample_uniform_distribution(vector, -1.0, 1.0, prec, seed);
  } else if (dist == PowTwo) {
    sample_uniform_distribution(vector, std::pow(2.0, pow2k),
                                std::pow(2.0, pow2k + 1), prec, seed);
  } else if (dist == Ones) {
    constant_initialization(vector, 1.0, prec);
  } else {
    throw std::invalid_argument("Invalid distribution");
  }
}

/* instantiate the templates */
template void sample_random_vector<double>(std::vector<double> &, Precision,
                                           Distribution, const int, const int);
template void sample_random_vector<float>(std::vector<float> &, Precision,
                                          Distribution, const int, const int);
template void sample_random_vector<half>(std::vector<half> &, Precision,
                                         Distribution, const int, const int);
