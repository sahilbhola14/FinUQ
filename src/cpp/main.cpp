#include <iostream>
#include <vector>

#include "definition.hpp"
#include "dot_product.hpp"
#include "gamma.hpp"
#include "utils.hpp"

/* gamma experiments */
void run_all_compare_gamma_experiments(Precision prec) {
  /* configuration */
  gamma_config gamma_cfg;
  gamma_cfg.prec = prec;
  /* vary confidence for uniform model*/
  gamma_cfg.bound_model = Uniform;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 2.01, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 2.001;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 2.01, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 2.01;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
  /* vary confidence for beta model (alpha = 2.2, beta=2.0)*/
  gamma_cfg.bound_model = Beta;
  gamma_cfg.beta_dist_alpha = 2.1;
  gamma_cfg.beta_dist_beta = 2.00;
  gamma_cfg.confidence = 0.9;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.95;
  compare_gamma(gamma_cfg);
  gamma_cfg.confidence = 0.99;
  compare_gamma(gamma_cfg);
}

/* dot product experiments */
void run_all_dot_product_experiments(Precision prec) {
  /* configuration */
  dot_product_config dot_product_cfg;

  dot_product_cfg.prec = prec;            // sampling precision
  dot_product_cfg.gamma_cfg.prec = prec;  // bound precision

  dot_product_cfg.num_experiments = 2;          // number of experiments
  dot_product_cfg.gamma_cfg.confidence = 0.90;  // overall confidence

  /* Data distribution: U(0,1) */
  dot_product_cfg.dist = ZeroOne;  // data distribution
  // uniform bound model
  /* dot_product_cfg.gamma_cfg.bound_model = Uniform; */
  /* run_dot_product_backward_error_experiment(dot_product_cfg); */
  /* run_dot_product_forward_error_experiment(2000000, dot_product_cfg); */

  /* // beta bound model */
  dot_product_cfg.gamma_cfg.bound_model = Beta;
  dot_product_cfg.gamma_cfg.beta_dist_alpha = 2.2;
  dot_product_cfg.gamma_cfg.beta_dist_beta = 2.0;
  run_dot_product_backward_error_experiment(dot_product_cfg);
  /* run_dot_product_forward_error_experiment(2000000, dot_product_cfg); */
}

int main(int argc, char **argv) {
  std::string experiment = "single_dot_product";
  if (argc > 1) {
    experiment = argv[1];
  }

  // Experiments
  if (experiment == "compare_gamma") {
    /* single precision */
    run_all_compare_gamma_experiments(Single);
    /* /1* single precision *1/ */
    /* run_all_compare_gamma_experiments(Half); */
  } else if (experiment == "single_dot_product") {
    run_all_dot_product_experiments(Single);
  }
  return 0;
}
