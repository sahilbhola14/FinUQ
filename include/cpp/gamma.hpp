#ifndef GAMMA_HPP
#define GAMMA_HPP

#include "definition.hpp"
#include <iostream>

/* configuration for the gamma */
struct gamma_config {
    Precision prec=Single; // Precision for computing the bounds
    double confidence=0.97; // confidence in the bounds
    BoundModel bound_model=Uniform; // bound model for variance-informed bounds
    double beta_dist_alpha = 2.01; // alpha parameter of the beta distribution
    double beta_dist_beta = 2.0; // beta parameter of the beta distribution
};

/* gamma results */
struct gamma_result {
    int n; // vector size
    long double gamma_det; // gamma deterministic
    long double gamma_mprea; // mean-informed probabilitic gamma
    long double gamma_vprea; // variance-informed probabilitic gamma
};


/* compare gamma values */
void compare_gamma(const gamma_config &gamma_cfg = gamma_config(), bool verbose = true);

/* get gamma */
gamma_result get_gamma(const int n, const gamma_config &gamma_cfg, const long double one_minus_zeta);

#endif
