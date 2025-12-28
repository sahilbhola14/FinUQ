#ifndef PROB_MODEL_HPP
#define PROB_MODEL_HPP

#include "definition.hpp"

struct log1pdeltastats {
    double mean;
    double var;
    double bound;
};

/* log1pdeltastats */
log1pdeltastats get_log1pdelta_stats(Precision prec, BoundModel bound_model, double beta_dist_alpha = 4.0, double beta_dist_beta = 2.0);


#endif
