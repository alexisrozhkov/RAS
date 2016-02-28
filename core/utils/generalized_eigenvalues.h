//
// Created by alexey on 28.02.16.
//

#ifndef RAS_GENERALIZED_EIGENVALUES_H
#define RAS_GENERALIZED_EIGENVALUES_H

#include <ras_types.h>

// using Eigen and real generalized Schur decomposition under the hood
// returns nominator and denominator for each eigenvalue (to be able to represent infinite eigenvalues to some extent)
void generalizedEigenvals(const Mat2D &A, const Mat2D &B, Mat2D &al, Mat2D &be);

#endif //RAS_GENERALIZED_EIGENVALUES_H
