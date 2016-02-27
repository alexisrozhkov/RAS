//
// Created by alexey on 21.02.16.
//

#include <gtest/gtest.h>
#include <balls_and_bins.h>

#include "balls_and_bins_data.h"


cv::Mat1i getExpectedMat(const int balls, const int bins) {
  assert(balls >= 1 && balls <= 3);
  assert(bins >= 1 && bins <= 3);

  auto expectedVec = expectedVals[balls-1][bins-1];
  int rows = int(expectedVec.size())/bins;

  cv::Mat1i expectMat = cv::Mat1i(expectedVec).t();  // todo: check why this transpose is needed
  return expectMat.reshape(0, rows);
}

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

class BallsAndBinsTest : public testing::TestWithParam<std::tuple<int, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};

void checkOutput(const cv::Mat1i& expected, const std::vector<cv::Mat1i>& result, const int expectNum) {
  EXPECT_EQ(result.size(), expectNum);
  EXPECT_TRUE(isIntMatrixEqual(result.back(), expected));
}

INSTANTIATE_TEST_CASE_P(First33Arguments,
                        BallsAndBinsTest,
                        ::testing::Combine(testing::Range(1, 4),
                                           testing::Range(1, 4)));

TEST_P(BallsAndBinsTest, checkResult)
{
  unsigned int balls = (uint)std::get<0>(GetParam()),
               bins =  (uint)std::get<1>(GetParam());

  cv::Mat1i expectMat = getExpectedMat(balls, bins);

  checkOutput(expectMat, balls_and_bins(balls, bins), 1);
  checkOutput(expectMat, balls_and_bins(balls, bins, true), balls);
}