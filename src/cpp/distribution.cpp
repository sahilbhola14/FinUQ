#include "distribution.hpp"

#include <cuda_fp16.h>

#include <cassert>
#include <iostream>
#include <stdexcept>

/* sample from N(0,1) */
template <typename T>
void sample_normal_vector(std::vector<T> &vector, std::mt19937 &gen) {
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
                                 const double upper, std::mt19937 &gen) {
  /* uniform distribution (double)*/
  std::uniform_real_distribution<double> uniform(lower, upper);
  /* sample random vector */
  for (auto &x : vector) {
    x = static_cast<T>(uniform(gen));
  }
}

/* sample from Beta distribution(alpha,beta) */
template <typename T>
void sample_beta_distribution(std::vector<T> &vector, const double alpha,
                              const double beta, std::mt19937 &gen) {
  /* gamma distribution */
  assert(alpha > 0 && "alpha value must be strictly positive");
  assert(beta > 0 && "beta value must be strictly positive");
  std::gamma_distribution<double> dist_a(alpha, 1.0);
  std::gamma_distribution<double> dist_b(beta, 1.0);
  /* sample random vector */
  for (auto &x : vector) {
    double a = dist_a(gen);
    double b = dist_b(gen);
    x = static_cast<T>(a / (a + b));
  }
}

/* constnat initialization */
template <typename T>
void constant_initialization(std::vector<T> &vector, const double constant) {
  /* initialize */
  for (auto &x : vector) {
    x = static_cast<T>(constant);
  }
}

/* sample random vector */
template <typename T>
void sample_random_vector(std::vector<T> &vector, Precision prec,
                          Distribution dist, std::mt19937 &gen,
                          const int pow2k) {
  if (dist == Normal) {
    sample_normal_vector(vector, gen);
  } else if (dist == ZeroOne) {
    sample_uniform_distribution(vector, 0.0, 1.0, gen);
  } else if (dist == MinusOnePlusOne) {
    sample_uniform_distribution(vector, -1.0, 1.0, gen);
  } else if (dist == PowTwo) {
    sample_uniform_distribution(vector, std::pow(2.0, pow2k),
                                std::pow(2.0, pow2k + 1), gen);
  } else if (dist == Ones) {
    constant_initialization(vector, 1.0);
  } else {
    throw std::invalid_argument("Invalid distribution");
  }
}

/* instantiate the templates */
template void sample_random_vector<double>(std::vector<double> &, Precision,
                                           Distribution, std::mt19937 &,
                                           const int);
template void sample_random_vector<float>(std::vector<float> &, Precision,
                                          Distribution, std::mt19937 &,
                                          const int);
template void sample_random_vector<half>(std::vector<half> &, Precision,
                                         Distribution, std::mt19937 &,
                                         const int);
