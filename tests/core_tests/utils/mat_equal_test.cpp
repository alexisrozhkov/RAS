//
// Created by alexey on 27.02.16.
//

#include "mat_equal_test.h"

const double comparisonEpsilon = 1e-10;

testing::AssertionResult isIntMatrixEqual(const cv::Mat1i& a, const cv::Mat1i& b) {
  if(a.type() != b.type()) {
    return testing::AssertionFailure() << testing::Message("Matrix type mismatch");
  }

  if((a.cols != b.cols) || (a.rows != b.rows)) {
    return testing::AssertionFailure() << testing::Message("Matrix size mismatch");
  }

  for(int j = 0; j < a.rows; j++) {
    for(int i = 0; i < a.cols; i++) {
      if(a(j, i) != b(j, i)) {
        return testing::AssertionFailure() << testing::Message("Matrix content mismatch");
      }
    }
  }

  return testing::AssertionSuccess();
}

testing::AssertionResult isDblMatrixEqual(const Mat2D& a, const Mat2D& b) {
  if(a.type() != b.type()) {
    return testing::AssertionFailure() << testing::Message("Matrix type mismatch");
  }

  if((a.cols != b.cols) || (a.rows != b.rows)) {
    return testing::AssertionFailure() << testing::Message("Matrix size mismatch");
  }

  for(int j = 0; j < a.rows; j++) {
    for(int i = 0; i < a.cols; i++) {
      if((std::isnan(a(j, i))) || (std::isnan(b(j, i))) || (std::fabs(a(j, i) - b(j, i)) > comparisonEpsilon)) {
        return testing::AssertionFailure() << testing::Message("Matrix content mismatch: ")
            << a(j, i) << testing::Message(" != ") <<  b(j, i);
      }
    }
  }

  return testing::AssertionSuccess();
}