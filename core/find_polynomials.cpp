//
// Created by alexey on 27.02.16.
//

#include <generalized_eigenvalues.h>

#include "find_polynomials.h"


Mat2D find_polynomials(const Mat2D &data,
                       const Mat3D &derivative,
                       const FindPolyMethod method,
                       const int charDimension) {
  if(method == FISHER) {
    const EmbValT RAYLEIGHQUOTIENT_EPSILON = 10;

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

      Mat2D temp_ = temp * temp.t();
      B += temp_;
    }

    // according to the paper denominator should be regularized.
    // but in the reference implementation nominator is.
    // todo: check this more closely or maybe contact the author
    A += RAYLEIGHQUOTIENT_EPSILON*Mat2D::eye(veroneseDimension, veroneseDimension);

    Mat2D alphas, betas;
    generalizedEigenvals(A, B, alphas, betas);

    // find eigenvalues from ratios and sort
    // perhaps special care has to be taken for cases when beta ~ 0
    std::vector<EmbValT> eigenvals((uint)alphas.rows);
    for(int i = 0; i < alphas.rows; i++) {
      eigenvals[i] = fabs(alphas(i))/fabs(betas(i));
    }
    std::sort(eigenvals.begin(), eigenvals.end());

    Mat2D out(veroneseDimension, charDimension);
    for(int i = 0; i < charDimension; i++) {
      // given an eigenval, we can easily find corresponding eigenvector
      cv::SVD::solveZ(A - eigenvals[i]*B, out.col(i));

      // no need to normalize since cv::SVD::solveZ seeks for solution with norm = 1
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