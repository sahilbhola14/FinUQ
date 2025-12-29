#include <iostream>
#include <vector>

#include "gamma.hpp"
/* #include "definition.hpp" */
#include "distribution.hpp"

int main(int argc, char **argv) {
  std::string experiment = "compare_gamma";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    compare_gamma();
  } else if (experiment == "single_dot_product") {
    std::vector<double> vector(5);
    sample_random_vector(vector, Single, Ones);
  }
  return 0;
}
