#ifndef MATRIX_VECTOR_HPP
#define MATRIX_VECTOR_HPP

#include "definition.hpp"
#include "gamma.hpp"
#include "dot_product.hpp"

/* configuration for matrix-vector products*/
struct matvec_product_config {
  Precision prec = Single; // precision for sampling random vectors
  Distribution dist=Normal; // distribution for the random vectors
  int num_experiments = 100; // number of experiments
  gamma_config gamma_cfg; // bounds config
};

/* /1* dot product experiment *1/ */
void run_matrix_vector_product_backward_error_experiment(const matvec_product_config &matvec_product_cfg);

/* /1* experiments *1/ */
/* void run_all_matrix_vector_product_experiments(Precision prec); */


#endif
