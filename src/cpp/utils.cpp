#include "utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int getGridSize(int blockSize, int N) {
  int gridSize = (N + blockSize - 1) / blockSize;
  return gridSize;
}

double computeUnitRoundOff(Precision prec) {
  double base, precision;
  double urd = 0.0;
  if (prec == Half) {
    base = 2.0;
    precision = 11.0;
  } else if (prec = Float) {
    base = 2.0;
    precision = 24.0;
  } else if (prec = Double) {
    base = 2.0;
    precision = 53.0;
  } else {
    printf("Error: Invalid precision");
    assert(false);
  }
  urd = pow(base, -(precision - 1.0)) / 2.0;
  return urd;
}
