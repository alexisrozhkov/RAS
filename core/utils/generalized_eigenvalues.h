// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_GENERALIZED_EIGENVALUES_H_
#define CORE_UTILS_GENERALIZED_EIGENVALUES_H_

#include <utils/ras_types.h>

// using Eigen and real generalized Schur decomposition under the hood
// returns nominator and denominator for each eigenvalue
// (to be able to represent infinite eigenvalues to some extent)
void generalizedEigenvals(const Mat2D &A, const Mat2D &B, Mat2D *al, Mat2D *be);

#endif  // CORE_UTILS_GENERALIZED_EIGENVALUES_H_
