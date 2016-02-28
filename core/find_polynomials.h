//
// Created by alexey on 27.02.16.
//

#ifndef RAS_FIND_POLYNOMIALS_H
#define RAS_FIND_POLYNOMIALS_H

#include <ras_types.h>


enum FindPolyMethod {
  FISHER,
  LLE
};


Mat2D find_polynomials(const Mat2D &data,
                       const Mat3D &derivative,
                       const FindPolyMethod method,
                       const int charDimension=1);

#endif //RAS_FIND_POLYNOMIALS_H
