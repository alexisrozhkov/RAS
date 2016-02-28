//
// Created by alexey on 27.02.16.
//

#include <iostream>
#include <generalized_eigenvalues.h>

#include "find_polynomials.h"


Mat2D find_polynomials(const Mat2D &data,
                       const Mat3D &derivative,
                       const FindPolyMethod method,
                       const int charDimension) {
  if(method == FISHER) {
    EmbValT RAYLEIGHQUOTIENT_EPSILON = 10;

    const int veroneseDimension = (uint)derivative.size(),
              dimensionCount = derivative[0].rows,
              sampleCount = derivative[0].cols;

    Mat2D B = Mat2D::zeros(veroneseDimension, veroneseDimension);
    Mat2D A = data * data.t();

    for(int dimensionIndex = 0; dimensionIndex < dimensionCount; dimensionIndex++) {
      Mat2D temp(veroneseDimension, sampleCount);
      for(int j = 0; j < veroneseDimension; j++) {
        for(int i = 0; i < sampleCount; i++) {
          temp(j, i) = derivative[j](dimensionIndex, i);
        }
      }

      B += temp * temp.t();
    }

    A += RAYLEIGHQUOTIENT_EPSILON*Mat2D::eye(veroneseDimension, veroneseDimension);

    Mat2D al, be;
    generalizedEigenvals(A, B, al, be);
    //std::cout << al << std::endl;
    //std::cout << be << std::endl;
    // not sure if element-wise al/be ratios are sorted or not... todo: check
    Mat2D out(veroneseDimension, charDimension);

    for(int i = 0; i < charDimension; i++) {
      const EmbValT alpha = al(al.rows-1-i);
      const EmbValT beta = be(be.rows-1-i);

      // given an eigenval, we can easily find corresponding eigenvector
      // todo: check if we can multiply A by beta instead of dividing B
      cv::SVD::solveZ(A - alpha*B/beta, out.col(i));

      out.col(i) /= cv::norm(out.col(i));
    }

    return out;
  }

  else {
    Mat2D w, u, vt;
    cv::SVD::compute(data, w, u, vt, cv::SVD::FULL_UV);

    Mat2D out(u.rows, charDimension);
    for(int j = 0; j < charDimension; j++) {
      u.col(u.cols-1-j).copyTo(out.col(charDimension-1-j));
    }

    return out;
  }
}