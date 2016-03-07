// Copyright 2016 Alexey Rozhkov

#ifndef CORE_FIND_POLYNOMIALS_H_
#define CORE_FIND_POLYNOMIALS_H_

#include <core/utils/mat_nd.h>


enum class FindPolyMethod {
  FISHER,
  LLE
};

// todo: add tests
Mat2D find_polynomials(const Mat2D &data,
                       const Mat3D &derivative,
                       const FindPolyMethod method,
                       const int charDimension = 1,
                       const int ignoreSample = -1);

#endif  // CORE_FIND_POLYNOMIALS_H_
