//
// Created by alexey on 27.02.16.
//

#include <gtest/gtest.h>
#include <mat_equal_test.h>
#include <perspective_embedding.h>
#include <find_polynomials.h>

#include "find_polynomials_data.h"


class FindPolynomialsTest : public testing::TestWithParam<std::tuple<int, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

TEST(FindPolynomialsTest, checkRes) {
  const int cols = 9;
  const int rows = int(outputBetterConditioned.size())/cols;

  const Mat2D expected = Mat2D(outputBetterConditioned).reshape(0, rows);
  const Mat2D inputMat = Mat2D(inputBetterConditioned).reshape(0, Kconst);

  auto e = perspective_embedding(inputMat, 1);

  auto result = find_polynomials(e.getV().back(),
                                 e.getD().back(),
                                 FISHER,
                                 cols);

  EXPECT_EQ(result.size(), expected.size());

  // handle the eigenvec 'sign' ambiguity
  for(int i = 0; i < result.cols; i++) {
    int sign = sgn(expected(0, i)) == sgn(result(0, i)) ? 1 : -1;

    // bigger epsilon since the mantissa in outputBetterConditioned is specified only up to 6th digit
    EXPECT_TRUE(isDblMatrixEqual(sign*result.col(i), expected.col(i), 1e-5));
  }
}