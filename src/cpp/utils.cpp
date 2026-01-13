#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>

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
          "gamma_mprea,gamma_vprea\n";
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
         << r.backward_error_bound.gamma_vprea << "\n";
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

/* load matrix market data */
/* std::vector<Matrix> loadMatrices(const std::string& filename) { */
/*     std::ifstream file(filename); */
/*     nlohmann::json j; */
/*     file >> j; */

/*     std::vector<Matrix> matrices; */
/*     for (auto& m : j["matrices"]) { */
/*         Matrix mat; */
/*         mat.id = m["id"]; */
/*         mat.name = m["name"]; */
/*         mat.rows = m["rows"]; */
/*         mat.cols = m["cols"]; */
/*         mat.data = m["data"].get<std::vector<double>>(); */
/*         matrices.push_back(mat); */
/*     } */
/*     return matrices; */
/* } */

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
