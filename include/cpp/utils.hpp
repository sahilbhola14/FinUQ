#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include "definition.hpp"
#include "gamma.hpp"
#include "backward_error.hpp"
#include "forward_error.hpp"

/* vector statistics */
struct vector_stats {
  double min; // minimum value of the vector
  double max; // max value of the vector
  double mean; // mean value of the vector
};

/* matrix struct */
template <typename T>
struct Matrix {
  size_t rows; // number of rows
  size_t cols; // number of cols
  size_t nnz; // number of non-zero elemens
  std::vector<T> data; // row-major
};

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
/* convert distribution to string for printing */
inline std::string to_string(Distribution dist){
    switch(dist){
        case Normal: return "N(0,1)";
        case ZeroOne: return "U(0,1)";
        case MinusOnePlusOne: return "U(-1,1)";
        case PowTwo: return "U(1,2)";
        case Ones: return "All ones";
        default: return "unknown";
    }
}

/* save */
void write_gamma_results_csv(const std::vector<gamma_result> &results, const std::string filename, bool verbose=false);
void write_backward_error_results_csv(const std::vector<backward_error_result> &results, const std::string filename, bool verbose=false);
void write_forward_error_results_csv(const forward_error_result &results, const std::string filename, bool verbose=false);
/* vector utils */
template <typename T>
void convert_vector_to_double(const std::vector<T> &source, std::vector<double> &target);
template <typename T>
void convert_vector_to_float(const std::vector<T> &source, std::vector<float> &target);
template <typename T>
void convert_vector_to_half(const std::vector<T> &source, std::vector<half> &target);
template <typename T>
void convert_vector_to_absolute_double(const std::vector<T> &source, std::vector<double> &target);
template <typename T>
void absolute_vector(const std::vector<T> &source, std::vector<T> &target);
vector_stats get_vector_stats(const std::vector<double>& v, bool verbose=false);
std::vector<int> make_logspace(int n_min, int n_max, int num_points);
std::vector<double> make_linspace(double start, double end, std::size_t num_points);
/* matrix utils */
std::vector<Matrix<double>> load_matrices_bin(const std::string& filename);
std::vector<Matrix<double>> get_matrix_market_data(std::string filename="square_matrices.bin");
template <typename T>
void copy_matrix_and_convert_precision(const Matrix<double> &source, Matrix<T> &target);

/* template initialization */
template struct Matrix<double>;
template struct Matrix<float>;
template struct Matrix<half>;

#endif
