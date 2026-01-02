#ifndef GAMMA_HPP
#define GAMMA_HPP

#include "definition.hpp"
#include <iostream>

/* configuration for the gamma */
struct gamma_config {
    Precision prec=Single; // Precision for computing the bounds
    double confidence=0.9; // confidence in the bounds
    BoundModel bound_model=Uniform; // bound model for variance-informed bounds
    double beta_dist_alpha = 2.01; // alpha parameter of the beta distribution
    double beta_dist_beta = 2.0; // beta parameter of the beta distribution
};

/* gamma results */
struct gamma_result {
    int n; // vector size
    double gamma_det; // gamma deterministic
    double gamma_mprea; // mean-informed probabilitic gamma
    double gamma_vprea; // variance-informed probabilitic gamma
};


/* compare gamma values */
void compare_gamma(const gamma_config &cfg = gamma_config(), bool verbose = true);

/* get gamma */
gamma_result get_gamma(const int n, const gamma_config &cfg);

#endif
