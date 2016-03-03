// Copyright 2016 Alexey Rozhkov

#include <tests/core_tests/balls_and_bins_data.h>
#include <tests/core_tests/utils/mat_equal_test.h>
#include <core/utils/balls_and_bins.h>
#include <gtest/gtest.h>
#include <tuple>


cv::Mat1i getExpectedMat(const int balls, const int bins) {
  assert(balls >= 1 && balls <= 3);
  assert(bins >= 1 && bins <= 3);

  auto expectedVec = expectedVals[balls - 1][bins - 1];
  int rows = static_cast<int>(expectedVec.size()) / bins;

  // todo: check why this transpose is needed
  IndexMat2D expectMat = IndexMat2D(expectedVec).t();
  return expectMat.reshape(0, rows);
}

class BallsAndBinsTest: public testing::TestWithParam<std::tuple<int, int>> {
 public:
  virtual void SetUp() { }
  virtual void TearDown() { }
};

void checkOutput(const IndexMat2D &expected,
                 const IndexMat2DArray &result,
                 const uint expectNum) {
  EXPECT_EQ(result.size(), expectNum);
  EXPECT_TRUE(isIntMatrixEqual(result.back(), expected));
}

INSTANTIATE_TEST_CASE_P(First33Arguments,
                        BallsAndBinsTest,
                        ::testing::Combine(testing::Range(1, 4),
                                           testing::Range(1, 4)));

TEST_P(BallsAndBinsTest, checkResult) {
  unsigned int balls = (uint) std::get<0>(GetParam()),
      bins = (uint) std::get<1>(GetParam());

  IndexMat2D expectMat = getExpectedMat(balls, bins);

  checkOutput(expectMat, balls_and_bins(balls, bins), 1);
  checkOutput(expectMat, balls_and_bins(balls, bins, true), balls);
}
