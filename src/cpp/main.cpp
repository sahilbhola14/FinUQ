#include <iostream>
#include <vector>

#include "definition.hpp"
#include "dot_product.hpp"
#include "gamma.hpp"
#include "utils.hpp"

/* dot product experiments */
void run_all_dot_product_experimens(Precision prec) {
  /* configuration */
  dot_product_config dot_product_cfg;
  dot_product_cfg.prec = prec;            // sampling precision
  dot_product_cfg.num_experiments = 10;   // sampling precision
  dot_product_cfg.gamma_cfg.prec = prec;  // bound precision

  /* Data distribution: U(0,1) */
  dot_product_cfg.dist = ZeroOne;  // data distribution

  // uniform bound model
  dot_product_cfg.gamma_cfg.bound_model = Uniform;
  /* run_dot_product_backward_error_experiment(dot_product_cfg); // backward
   * error experiment */
  run_dot_product_forward_error_experiment(1000, dot_product_cfg);

  /* // beta bound model */
  /* dot_product_cfg.gamma_cfg.bound_model = Beta; */
  /* run_dot_product_backward_error_experiment(dot_product_cfg); */
}

int main(int argc, char **argv) {
  std::string experiment = "single_dot_product";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    compare_gamma();
  } else if (experiment == "single_dot_product") {
    run_all_dot_product_experimens(Single);
  }
  return 0;
}
