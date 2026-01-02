#ifndef PROB_MODEL_HPP
#define PROB_MODEL_HPP

#include "definition.hpp"

struct log1pdeltastats {
    double mean=0.0;
    double var=0.0;
    double bound=0.0;
};

/* log1pdeltastats */
log1pdeltastats get_log1pdelta_stats(Precision prec, BoundModel bound_model, double beta_dist_alpha = 4.0, double beta_dist_beta = 2.0, bool verbose=false);

/* sign of rounding error random variable */
void check_mean_rounding_error_sign(Precision prec, BoundModel bound_model, double beta_dist_alpha, double beta_dist_beta);

/* individual bound confidence */
double compute_individual_bound_zeta_confidence(const int arithmetic_operations, double total_confidence);

#endif
