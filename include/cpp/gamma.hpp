#ifndef GAMMA_HPP
#define GAMMA_HPP

#include "definition.hpp"
#include <iostream>

double getGamma(int N, Precision prec, BoundType btype, double confidence = 0.99);
void compareGamma(int N_lower, Precision prec, std::string filename, int bit_shift = 2, int max_shift = 12, double confidence = 0.99);

#endif
