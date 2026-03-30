#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

/* write gamma results to csv */
void write_gamma_results_csv(const std::vector<gamma_result> &results,
                             std::string filename, bool verbose) {
  /* open the file */
  std::ofstream file(filename);
  /* check */
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << filename << "\n";
    return;
  }
  /* header */
  file << "n,gamma_det,gamma_mprea,gamma_vprea\n";
  /* numerical font */
  file << std::scientific << std::setprecision(10);
  /* write the results */
  for (const auto &r : results) {
    file << std::left << std::setw(12) << r.n << ", " << std::setw(18)
         << r.gamma_det << ", " << std::setw(18) << r.gamma_mprea << ", "
         << std::setw(18) << r.gamma_vprea << "\n";
  }
  /* close the file */
  file.close();
  /* print */
  if (verbose == true) {
    std::cout << "gamma results saved to " << filename << std::endl;
  }
}

/* write backward error results to csv */
void write_backward_error_results_csv(
    const std::vector<backward_error_result> &results, std::string filename,
    bool verbose) {
  /* open the file */
  std::ofstream file(filename);
  /* check */
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << filename << "\n";
    return;
  }
  /* header */
  file << "n,backward_error_min,backward_error_max,backward_error_mean,gamma_"
          "det,"
          "gamma_mprea,gamma_vprea,nnz_to_size_ratio\n";
  /* numerical font */
  file << std::scientific << std::setprecision(10);
  /* write the results */
  for (const auto &r : results) {
    file << std::left << std::setw(12) << r.n << ", " << std::setw(18)
         << r.backward_error_min << ", " << std::setw(18)
         << r.backward_error_max << ", " << std::setw(18)
         << r.backward_error_mean << ", " << std::setw(18)
         << r.backward_error_bound.gamma_det << ", " << std::setw(18)
         << r.backward_error_bound.gamma_mprea << ", " << std::setw(18)
         << r.backward_error_bound.gamma_vprea << ", " << std::setw(18)
         << r.nnz_to_size_ratio << "\n";
  }
  /* close the file */
  file.close();
  /* print */
  if (verbose == true) {
    std::cout << "backward error results saved to " << filename << std::endl;
  }
}

/* write forward error results to csv */
void write_forward_error_results_csv(const forward_error_result &results,
                                     std::string filename, bool verbose) {
  /* open the file */
  std::ofstream file(filename);
  /* check */
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << filename << "\n";
    return;
  }
  /* header */
  file << "n,forward_error,forward_error_model,gamma_det,"
          "gamma_mprea,gamma_vprea\n";
  /* numerical font */
  file << std::scientific << std::setprecision(10);
  /* write the results */
  for (int i = 0; i < results.forward_error.size(); i++) {
    file << std::left << std::setw(12) << results.n << ", " << std::setw(18)
         << results.forward_error[i] << ", " << std::setw(18)
         << results.forward_error_model[i] << ", " << std::setw(18)
         << results.forward_error_bound[i].gamma_det << ", " << std::setw(18)
         << results.forward_error_bound[i].gamma_mprea << ", " << std::setw(18)
         << results.forward_error_bound[i].gamma_vprea << "\n";
  }
  /* close the file */
  file.close();
  /* print */
  if (verbose == true) {
    std::cout << "forward error results saved to " << filename << std::endl;
  }
}

/* write bvp forward error results to csv */
void write_bvp_forward_error_results_csv(
    const std::vector<bvp_forward_error_result> &results, std::string filename,
    bool verbose) {
  /* open the file */
  std::ofstream file(filename);
  /* check */
  if (!file.is_open()) {
    std::cerr << "Error: could not open file " << filename << "\n";
    return;
  }
  /* header */
  file << "n,forward_error,forward_error_model,gamma_det,"
          "gamma_mprea,gamma_vprea\n";
  /* numerical font */
  file << std::scientific << std::setprecision(10);
  /* write the results */
  for (const auto &r : results) {
    file << std::left << std::setw(12) << r.n << ", " << std::setw(18)
         << r.qoi_forward_error << ", " << std::setw(18)
         << r.qoi_forward_error_model << ", " << std::setw(18)
         << r.forward_error_bound.gamma_det << ", " << std::setw(18)
         << r.forward_error_bound.gamma_mprea << ", " << std::setw(18)
         << r.forward_error_bound.gamma_vprea << "\n";
  }
  /* close the file */
  file.close();
  /* print */
  if (verbose == true) {
    std::cout << "forward error results saved to " << filename << std::endl;
  }
}

/* convert to double */
template <typename T>
void convert_vector_to_double(const std::vector<T> &source,
                              std::vector<double> &target) {
  target.clear();
  target.reserve(source.size());

  for (const T &x : source) {
    target.push_back(static_cast<double>(x));
  }
}

/* convert to float */
template <typename T>
void convert_vector_to_float(const std::vector<T> &source,
                             std::vector<float> &target) {
  target.clear();
  target.reserve(source.size());

  for (const T &x : source) {
    target.push_back(static_cast<float>(x));
  }
}

/* convert to half */
template <typename T>
void convert_vector_to_half(const std::vector<T> &source,
                            std::vector<half> &target) {
  target.clear();
  target.reserve(source.size());

  for (const T &x : source) {
    target.push_back(static_cast<half>(x));
  }
}

/* absolute value*/
template <typename T>
void absolute_vector(const std::vector<T> &source, std::vector<T> &target) {
  target.clear();
  target.reserve(source.size());

  for (const T &x : source) {
    target.push_back(std::abs(x));
  }
}

/* take absolute and convert to double */
template <typename T>
void convert_vector_to_absolute_double(const std::vector<T> &source,
                                       std::vector<double> &target) {
  target.clear();
  target.reserve(source.size());

  for (const T &x : source) {
    target.push_back(std::abs(static_cast<double>(x)));
  }
}

/* compute vector statistics */
vector_stats get_vector_stats(const std::vector<double> &v, bool verbose) {
  vector_stats stats;

  if (v.empty()) throw std::runtime_error("get_vector_stats: vector is empty");

  auto minmax = std::minmax_element(v.begin(), v.end());

  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / static_cast<double>(v.size());

  stats.min = static_cast<double>(*minmax.first);
  stats.max = static_cast<double>(*minmax.second);
  stats.mean = mean;
  /* verbose */
  if (verbose == true) {
    std::cout << "vector min: " << stats.min << std::endl;
    std::cout << "vector max: " << stats.max << std::endl;
    std::cout << "vector mean: " << stats.mean << std::endl;
  }
  return stats;
}

/* make logspace */
std::vector<int> make_logspace(int n_min, int n_max, int num_points) {
  std::vector<int> vals;
  vals.reserve(num_points);

  double log_min = std::log10(n_min);
  double log_max = std::log10(n_max);

  for (int i = 0; i < num_points; ++i) {
    double t = static_cast<double>(i) / (num_points - 1);
    int v = static_cast<int>(
        std::round(std::pow(10.0, log_min + t * (log_max - log_min))));
    if (vals.empty() || v != vals.back()) vals.push_back(v);
  }
  return vals;
}

/* make linspace */
template <typename T>
std::vector<T> make_linspace(T start, T end, std::size_t num_points) {
  if (num_points == 0) {
    return {};
  }
  if (num_points == 1) {
    return {start};
  }

  std::vector<T> result(num_points);
  T step = (end - start) / static_cast<T>(num_points - 1);

  for (std::size_t i = 0; i < num_points; ++i) {
    result[i] = start + step * static_cast<T>(i);
  }

  return result;
}

/* load matrix stored in binary format */
std::vector<Matrix<double>> load_matrices_bin(const std::string &filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open file");
  }

  int32_t nmat;
  f.read(reinterpret_cast<char *>(&nmat), sizeof(int32_t));

  if (nmat <= 0) {
    throw std::runtime_error("Invalid number of matrices");
  }

  std::vector<Matrix<double>> matrices;
  matrices.reserve(nmat);

  for (int i = 0; i < nmat; ++i) {
    int32_t rows, cols, nnz;
    f.read(reinterpret_cast<char *>(&rows), sizeof(int32_t));
    f.read(reinterpret_cast<char *>(&cols), sizeof(int32_t));
    f.read(reinterpret_cast<char *>(&nnz), sizeof(int32_t));

    if (rows <= 0 || cols <= 0) {
      throw std::runtime_error("Invalid matrix shape");
    }

    Matrix<double> M;
    M.rows = rows;
    M.cols = cols;
    M.nnz = nnz;
    M.data.resize(static_cast<size_t>(rows) * cols);

    f.read(reinterpret_cast<char *>(M.data.data()),
           M.data.size() * sizeof(double));

    matrices.push_back(std::move(M));
  }

  return matrices;
}

/* load the matrix market data */
std::vector<Matrix<double>> get_matrix_market_data(std::string filename) {
  /* /1* for testing *1/ */
  /* std::vector<Matrix<double>> matrices(1); */
  /* for (auto &m : matrices) { */
  /*   m.rows = 5; */
  /*   m.cols = 5; */
  /*   m.nnz = 25; */
  /*   m.data.reserve(25); */
  /*   for (int i = 0; i < 25; i++) { */
  /*     m.data.push_back(1.0); */
  /*   } */
  /* } */
  /* return matrices; */

  std::cout << "Loading matrix market data from file : " << filename
            << std::endl;
  std::vector<Matrix<double>> matrices = load_matrices_bin(filename);
  return matrices;
}

/* copy the matrix to a target matrix and change the precision */
template <typename T>
void copy_matrix_and_convert_precision(const Matrix<double> &source,
                                       Matrix<T> &target) {
  /* copy the num rows, cols, and nnz */
  target.rows = source.rows;
  target.cols = source.cols;
  target.nnz = source.nnz;
  /* reserve the size for the data */
  target.data.reserve(source.rows * source.cols);
  for (const double &d : source.data) {
    target.data.push_back(static_cast<T>(d));
  }
}

/* initialize templates */
template void convert_vector_to_double<double>(const std::vector<double> &,
                                               std::vector<double> &);
template void convert_vector_to_double<float>(const std::vector<float> &,
                                              std::vector<double> &);
template void convert_vector_to_double<half>(const std::vector<half> &,
                                             std::vector<double> &);
template void convert_vector_to_absolute_double<double>(
    const std::vector<double> &, std::vector<double> &);
template void convert_vector_to_absolute_double<float>(
    const std::vector<float> &, std::vector<double> &);
template void convert_vector_to_absolute_double<half>(const std::vector<half> &,
                                                      std::vector<double> &);

template void copy_matrix_and_convert_precision<double>(
    const Matrix<double> &source, Matrix<double> &target);
template void copy_matrix_and_convert_precision<float>(
    const Matrix<double> &source, Matrix<float> &target);
template void copy_matrix_and_convert_precision<half>(
    const Matrix<double> &source, Matrix<half> &target);

template std::vector<double> make_linspace(double, double, std::size_t);
template std::vector<int> make_linspace(int, int, std::size_t);
