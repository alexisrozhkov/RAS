// Copyright 2016 Alexey Rozhkov

#include <tests/core_tests/find_polynomials_data.h>
#include <tests/core_tests/utils/mat_equal_test.h>
#include <core/perspective_embedding.h>
#include <core/find_polynomials.h>
#include <gtest/gtest.h>
#include <tuple>


class FindPolynomialsTest : public
                            testing::TestWithParam<std::tuple<int, int>> {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(First32Arguments,
                        FindPolynomialsTest,
                        ::testing::Combine(testing::Range(0, 3),
                                           testing::Range(0, 2)));

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

TEST_P(FindPolynomialsTest, checkRes) {
  const int sizeIdx = std::get<0>(GetParam());
  const int motionIdx = std::get<1>(GetParam());
  const auto expectedVec = expectedResult[motionIdx][sizeIdx];

  const int cols = 1;
  const int rows = static_cast<int>(expectedVec.size())/cols;

  const Mat2D expected = Mat2D(expectedVec, true).reshape(0, rows);
  const Mat2D inputMat = Mat2D(inputRandom[sizeIdx], true).reshape(0, Kconst);

  const auto e = perspective_embedding(inputMat, (uint)motionIdx+1);

  const auto result = find_polynomials(e.getV(),
                                       e.getD(),
                                       FindPolyMethod::FISHER,
                                       cols);

  EXPECT_EQ(result.size(), expected.size());

  // handle the eigenvec 'sign' ambiguity
  for (int i = 0; i < result.cols; i++) {
    int sign = sgn(expected(0, i)) == sgn(result(0, i)) ? 1 : -1;

    // bigger epsilon since the mantissa in expected result is
    // specified only up to 6th digit
    EXPECT_TRUE(isDblMatrixEqual(sign*result.col(i), expected.col(i), 1e-5));
  }
}
