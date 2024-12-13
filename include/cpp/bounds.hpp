#ifndef BOUNDS_HPP
#define BOUNDS_HPP

#include "definition.hpp"

double dotProductBackwardBound(int N, Precision prec, BoundType btype, double confidence = 0.99);
double matVecBackwardBound(int N, Precision prec, BoundType btype, double confidence = 0.99);

#endif
