#ifndef UTILS_HPP
#define UTILS_HPP

#include "definition.hpp"

int get_grid_size(int blockSize, int N);
double compute_unit_roundoff(Precision prec);
void linspace(double start, double end, int N, double *x);

#endif
