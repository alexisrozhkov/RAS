// Copyright 2016 Alexey Rozhkov

#include <core/utils/math.h>
#include <algorithm>


// taken from opencv/modules/ml/src/inner_functions.cpp, with minor modification
// for some reason it is not part of OpenCV public interface
Mat2D Cholesky(const Mat2D &A) {
  int dim = A.rows;
  Mat2D S(dim, dim);

  int i, j, k;

  for (i = 0; i < dim; i++) {
    for (j = 0; j < i; j++)
      S(i, j) = 0.f;

    EmbValT sum = 0.f;
    for (k = 0; k < i; k++) {
      EmbValT val = S(k, i);
      sum += val * val;
    }

    S(i, i) = std::sqrt(std::max(A(i, i) - sum, 0.));
    EmbValT ival = 1.f / S(i, i);

    for (j = i + 1; j < dim; j++) {
      sum = 0;
      for (k = 0; k < i; k++)
        sum += S(k, i) * S(k, j);

      S(i, j) = (A(i, j) - sum) * ival;
    }
  }

  return S;
}

Mat2D hat(const Mat2D &v) {
  return (Mat2D(3, 3) <<     0, -v(2),  v(1),
                          v(2),     0, -v(0),
                         -v(1),  v(0),    0);
}

Mat2D kron(const Mat2D &a, const Mat2D &b) {
  Mat2D out(a.rows*b.rows, a.cols*b.cols);

  for (int j = 0; j < a.rows; j++) {
    for (int i = 0; i < a.cols; i++) {
      out(cv::Rect(i*b.cols, j*b.rows, b.cols, b.rows)) = a(j, i)*b;
    }
  }

  return out;
}
