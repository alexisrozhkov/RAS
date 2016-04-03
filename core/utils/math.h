// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_MATH_H_
#define CORE_UTILS_MATH_H_

#include <core/utils/mat_nd.h>

// Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S
// todo: add tests
Mat2D Cholesky(const Mat2D &A);

// Skew-symmetric matrix from 3d vector
Mat2D hat(const Mat2D &v);

// Kroneker product
Mat2D kron(const Mat2D &a, const Mat2D &b);

#endif  // CORE_UTILS_MATH_H_
