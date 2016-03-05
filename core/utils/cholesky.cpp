// Copyright 2016 Alexey Rozhkov

#include <core/utils/cholesky.h>
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
