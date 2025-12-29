#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>
#include <string>
#include "definition.hpp"
#include "gamma.hpp"

/* convert precison to string for printing*/
inline std::string to_string(Precision prec){
    switch(prec){
        case Half: return "half";
        case Single: return "single";
        case Double: return "double";
        default: return "unknown";
    }
}

/* convert bound model to string for printing */
inline std::string to_string(BoundModel bound_model){
    switch(bound_model){
        case Uniform: return "uniform";
        case Beta: return "beta";
        default: return "unknown";
    }
}
/* unit roundoff */
double compute_unit_roundoff(Precision prec);
/* save */
void write_gamma_results_csv(const std::vector<gamma_result> &results, const std::string filename, bool verbose=false);

#endif
