// Copyright 2016 Alexey Rozhkov

#include <core/utils/subspace_angle.h>
#include <algorithm>


EmbValT subspace_angle(const Mat2D &A_, const Mat2D &B_) {
  Mat2D A = A_.clone(),
      B = B_.clone();

  if (A.cols < B.cols) {
    A = B_;
    B = A_;
  }

  // Compute the projection the most accurate way, according to [1].
  for (int k = 0; k < A.cols; k++) {
    B -= A.col(k)*(A.col(k).t()*B);
  }

  // Make sure it's magnitude is less than 1.
  return asin(std::min(1.0, (cv::norm(B))));
}
