// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_ARMA_WRAPPER_H_
#define CORE_UTILS_ARMA_WRAPPER_H_

#include <core/utils/mat_nd.h>


Mat2D SVD_S(const Mat2D &A);
Mat2D SVD_U(const Mat2D &A);
Mat2D SVD_V(const Mat2D &A);

void generalizedSchur(const Mat2D &A,
                      const Mat2D &B,
                      Mat2D *al,
                      Mat2D *be);

#endif  // CORE_UTILS_ARMA_WRAPPER_H_
