#ifndef UTILS_HPP
#define UTILS_HPP

#include "definition.hpp"

int getGridSize(int blockSize, int N);
double computeUnitRoundOff(Precision prec);
void linspace(double start, double end, int N, double *x);

#endif
