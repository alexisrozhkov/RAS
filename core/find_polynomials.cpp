//
// Created by alexey on 27.02.16.
//

#include <iostream>
#include "find_polynomials.h"


Mat2D find_polynomials(const Mat2D &data,
                       const Mat3D &derivative,
                       const FindPolyMethod method,
                       const int charDimension) {
  if(method == FISHER) {
    return Mat2D();
  }

  else {
    Mat2D w, u, vt;
    cv::SVD::compute(data, w, u, vt, cv::SVD::FULL_UV);

    //std::cout << w << std::endl;

    Mat2D out(u.rows, charDimension);
    for(int j = 0; j < charDimension; j++) {
      u.col(u.cols-1-j).copyTo(out.col(charDimension-1-j));
    }

    return out;
  }
}