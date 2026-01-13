#include "matrix_vector.hpp"

#include <iostream>

#include "utils.hpp"

void load_matrix_market_data() {
  std::string filename = "square_matrices.bin";
  std::vector<Matrix> matrices = load_matrices_bin(filename);
}
