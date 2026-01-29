#include <iostream>
#include <vector>

#include "boundary_value_prob.hpp"
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
  } else if (experiment == "matrix_market") {
    run_all_matrix_vector_product_experiments(Single);
    run_all_matrix_vector_product_experiments(Half);
  } else if (experiment == "ode") {
    run_all_ode_experiments(Single);
    run_all_ode_experiments(Half);
  } else if (experiment == "testing") {
    // run_all_ode_experiments(Single);
    run_all_ode_experiments(Half);
  }
  return 0;
}
