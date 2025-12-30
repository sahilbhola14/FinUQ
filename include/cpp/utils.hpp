#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>
#include <string>
#include <cuda_fp16.h>
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
/* vector utils */
template <typename T>
void convert_vector_to_double(const std::vector<T> &source, std::vector<double> &target);
template <typename T>
void convert_vector_to_float(const std::vector<T> &source, std::vector<float> &target);
template <typename T>
void convert_vector_to_half(const std::vector<T> &source, std::vector<half> &target);
/* absolute value */
template <typename T>
void absolute_vector(const std::vector<T> &source, std::vector<T> &target);

#endif
