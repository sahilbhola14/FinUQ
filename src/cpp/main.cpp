#include <iostream>
#include <vector>

#include "dot_product.hpp"
#include "gamma.hpp"
#include "utils.hpp"

int main(int argc, char **argv) {
  std::string experiment = "single_dot_product";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    compare_gamma();
  } else if (experiment == "single_dot_product") {
    dot_product_config dot_product_cfg;
    dot_product_cfg.prec = Single;  // sampling precision
    /* dot_product_cfg.gamma_cfg.prec = Half; // bound precision */
    run_dot_product_experiment(dot_product_cfg);
  }
  return 0;
}
