// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_CHOLESKY_H_
#define CORE_UTILS_CHOLESKY_H_

#include <core/utils/ras_types.h>

// Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S
// todo: add tests
Mat2D Cholesky(const Mat2D &A);

#endif  // CORE_UTILS_CHOLESKY_H_
