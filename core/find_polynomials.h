// Copyright 2016 Alexey Rozhkov

#ifndef CORE_FIND_POLYNOMIALS_H_
#define CORE_FIND_POLYNOMIALS_H_

#include <utils/ras_types.h>


enum class FindPolyMethod {
  FISHER,
  LLE
};

Mat2D find_polynomials(const Mat2D &data,
                       const Mat3D &derivative,
                       const FindPolyMethod method,
                       const int charDimension = 1);

#endif  // CORE_FIND_POLYNOMIALS_H_
