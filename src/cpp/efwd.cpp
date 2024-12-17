#include "efwd.hpp"

#include <cuda_fp16.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "ebwd.hpp"

void invertA(int N, double *sub_diag, double *main_diag, double *super_diag,
             double *abs_prod, double *efwd) {
  const int N_inner = N - 1;

  // Create a sparse matrix of size N_inner x N_inner
  Eigen::SparseMatrix<double> mat(N_inner, N_inner);

  // Set up a list of triplets to represent the non-zero values in the sparse
  // matrix
  std::vector<Eigen::Triplet<double>> triplets;

  // First row
  triplets.push_back(
      Eigen::Triplet<double>(0, 0, main_diag[0]));  // (0, 0) = main_diag[0]
  triplets.push_back(
      Eigen::Triplet<double>(0, 1, super_diag[0]));  // (0, 1) = super_diag[0]

  // Middle rows
  for (int ii = 1; ii < N_inner - 1; ii++) {
    triplets.push_back(Eigen::Triplet<double>(
        ii, ii, main_diag[ii]));  // (ii, ii) = main_diag[ii]
    triplets.push_back(Eigen::Triplet<double>(
        ii, ii - 1, sub_diag[ii]));  // (ii, ii-1) = sub_diag[ii]
    triplets.push_back(Eigen::Triplet<double>(
        ii, ii + 1, super_diag[ii]));  // (ii, ii+1) = super_diag[ii]
  }

  // Last row
  triplets.push_back(Eigen::Triplet<double>(
      N_inner - 1, N_inner - 1,
      main_diag[N_inner -
                1]));  // (N_inner-1, N_inner-1) = main_diag[N_inner-1]
  triplets.push_back(Eigen::Triplet<double>(
      N_inner - 1, N_inner - 2,
      sub_diag[N_inner - 1]));  // (N_inner-1, N_inner-2) = sub_diag[N_inner-1]

  // Set the matrix from the triplets
  mat.setFromTriplets(triplets.begin(), triplets.end());

  // Print the sparse matrix
  /* std::cout << "Sparse Matrix:\n" << mat << std::endl; */

  // Solve using SparseLU (direct solver)
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(mat);
  if (solver.info() != Eigen::Success) {
    std::cerr << "Decomposition failed!" << std::endl;
    return;
  }

  // Create an identity matrix
  Eigen::SparseMatrix<double> identity(N_inner, N_inner);
  identity.setIdentity();  // Set the matrix to identity

  // Solve for the inverse (A * X = I, where I is the identity)
  Eigen::SparseMatrix<double> mat_inv = solver.solve(identity);
  Eigen::MatrixXd mat_inv_dense = Eigen::MatrixXd(mat_inv);

  // Multiply
  int rows = mat_inv_dense.rows();
  int cols = mat_inv_dense.cols();
  double error = 0.0;

  for (int ii = 0; ii < rows; ii++) {
    double summation = 0.0;
    for (int jj = 0; jj < cols; jj++) {
      summation += std::abs(mat_inv_dense(ii, jj)) * abs_prod[jj];
    }
    error = std::max(error, summation);
  }

  *efwd = error;
}

template <typename T>
void computeForwardErrorThomas(int N, double *sub_diag, double *main_diag,
                               double *super_diag, double *rhs, T *a, T *b,
                               T *u, double *ebwd, double *efwd) {
  const int N_inner = N - 1;
  double *abs_prod = static_cast<double *>(malloc(N_inner * sizeof(double)));
  // Compute the absolute LU*u
  computeAbsoluteLUTimesSol(N, a, b, super_diag, u, abs_prod);

  invertA(N, sub_diag, main_diag, super_diag, abs_prod, efwd);
  *efwd = *ebwd * *efwd;
}

// Template compilation
template void computeForwardErrorThomas(int, double *, double *, double *,
                                        double *, float *, float *, float *,
                                        double *, double *);
template void computeForwardErrorThomas(int, double *, double *, double *,
                                        double *, half *, half *, half *,
                                        double *, double *);
