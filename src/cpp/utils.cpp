#include "utils.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>

/* compute the unit roundoff */
double compute_unit_roundoff(Precision prec) {
  double base, precision;
  double urd = 0.0;
  if (prec == Half) {
    base = 2.0;
    precision = 11.0;
  } else if (prec == Single) {
    base = 2.0;
    precision = 24.0;
  } else if (prec == Double) {
    base = 2.0;
    precision = 53.0;
  } else {
    printf("Error: Invalid precision");
    assert(false);
  }
  urd = pow(base, -(precision - 1.0)) / 2.0;
  return urd;
}

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
    file << std::left << std::setw(12) << r.n << std::setw(18) << r.gamma_det
         << std::setw(18) << r.gamma_mprea << std::setw(18) << r.gamma_vprea
         << "\n";
  }
  /* close the file */
  file.close();
  /* print */
  if (verbose == true) {
    std::cout << "gamma results saved to " << filename << std::endl;
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
void convert_vector_to_float(const std::vector<T> &source,
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
