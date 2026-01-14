#include <iostream>
#include <vector>

#include "definition.hpp"
#include "dot_product.hpp"
#include "gamma.hpp"
#include "matrix_vector.hpp"
#include "utils.hpp"

int main(int argc, char **argv) {
  /* std::string experiment = "compare_gamma"; */
  std::string experiment = "testing";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    run_all_compare_gamma_experiments(Single);
    run_all_compare_gamma_experiments(Half);
  } else if (experiment == "dot_product") {
    run_all_dot_product_experiments(Single);
    run_all_dot_product_experiments(Half);
  } else if (experiment == "testing") {
    matvec_product_config matvec_product_cfg;
    matvec_product_cfg.dist = Ones;
    Precision compute_prec = Half;
    matvec_product_cfg.prec = compute_prec;
    matvec_product_cfg.gamma_cfg.prec = compute_prec;
    matvec_product_cfg.num_experiments = 1;
    run_matrix_vector_product_backward_error_experiment(matvec_product_cfg);
  }
  return 0;
}
