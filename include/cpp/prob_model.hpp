#ifndef PROB_MODEL_HPP
#define PROB_MODEL_HPP

#include "definition.hpp"

struct log1pdeltastats {
    long double mean=0.0;
    long double var=0.0;
    long double bound=0.0;
};

/* log1pdeltastats */
log1pdeltastats get_log1pdelta_stats(Precision prec, BoundModel bound_model, double beta_dist_alpha = 4.0, double beta_dist_beta = 2.0, bool verbose=false);

/* sign of rounding error random variable */
void check_mean_rounding_error_sign(Precision prec, BoundModel bound_model, double beta_dist_alpha, double beta_dist_beta);

/* individual bound confidence */
long double compute_individual_bound_one_minus_zeta(const int number_of_bounds, double total_confidence);

#endif
