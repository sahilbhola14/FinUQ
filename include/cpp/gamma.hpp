#ifndef GAMMA_HPP
#define GAMMA_HPP

#include "definition.hpp"
#include <iostream>

struct gamma_config {
    Precision prec=Single;
    int n_lower=10;
    int n_mults=3;
    float confidence=0.99; // confidence in the bounds
    BoundModel bound_model=Uniform; // bound model for variance-informed bounds
    double beta_dist_alpha = 4.0; // alpha parameter of the beta distribution
    double beta_dist_beta = 2.0; // beta parameter of the beta distribution
};

void compare_gamma(const gamma_config &cfg = gamma_config());

#endif
