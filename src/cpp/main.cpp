#include <iostream>
#include <vector>

#include "dot_product.hpp"
#include "gamma.hpp"

int main(int argc, char **argv) {
  std::string experiment = "single_dot_product";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    compare_gamma();
  } else if (experiment == "single_dot_product") {
    run_dot_product_experiment(Single, Ones);
  }
  return 0;
}
