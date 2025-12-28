#include <iostream>

#include "gamma.hpp"

int main(int argc, char **argv) {
  std::string experiment = "compare_gamma";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    compare_gamma();
  }
  return 0;
}
