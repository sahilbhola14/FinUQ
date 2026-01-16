#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <vector>
#include <random>
#include "definition.hpp"

/* sample random vector in a given precision*/
template <typename T>
void sample_random_vector(std::vector<T> &vector, Precision prec, Distribution dist, std::mt19937 &gen, const int pow2k=0);

/* sample from U(a,b) */
template <typename T>
void sample_uniform_distribution(std::vector<T> &vector, const double lower,
                                 const double upper, std::mt19937 &gen);

#endif
