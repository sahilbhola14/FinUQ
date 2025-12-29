#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <vector>
#include "definition.hpp"

/* sample random vector in a given precision*/
template <typename T>
void sample_random_vector(std::vector<T> &vector, Precision prec, Distribution dist, const int pow2k=0, const int seed=42);
#endif
