//
// Created by alexey on 27.02.16.
//

#include <gtest/gtest.h>
#include <mat_equal_test.h>
#include <find_polynomials.h>

#include "find_polynomials_data.h"


class FindPolynomialsTest : public testing::TestWithParam<std::tuple<int, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};

INSTANTIATE_TEST_CASE_P(First22Arguments,
                        FindPolynomialsTest,
                        ::testing::Combine(testing::Range(0, 2),
                                           testing::Range(0, 2)));


TEST_P(FindPolynomialsTest, checkRes) {
  unsigned int zIdx = (uint)std::get<0>(GetParam()),
               nIdx =  (uint)std::get<1>(GetParam());

  const int cols = 9;
  const int rows = int(oneMotion[zIdx][nIdx].size())/cols;
  const Mat2D expected = Mat2D(oneMotion[zIdx][nIdx]).reshape(0, rows);

  const Mat2D inputMat = Mat2D(input[zIdx][nIdx]).reshape(0, Kconst);

  auto e = perspective_embedding(inputMat, 1);

  auto result = find_polynomials(e.getV().back(),
                                 e.getD().back(),
                                 FISHER,
                                 cols);

  std::cout << expected << std::endl;

  EXPECT_EQ(result.size(), expected.size());
  EXPECT_TRUE(isDblMatrixEqual(result, expected));
}