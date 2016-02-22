//
// Created by alexey on 21.02.16.
//

#include "gtest/gtest.h"
#include <balls_and_bins.h>


bool isIntMatrixEqual(const cv::Mat1i& a, const cv::Mat1i& b) {
  if(a.type() != b.type()) return 0;
  if((a.cols != b.cols) || (a.rows != b.rows)) return 0;

  for(int j = 0; j < a.rows; j++) {
    for(int i = 0; i < a.cols; i++) {
      if(a(j, i) != b(j, i)) return 0;
    }
  }

  return 1;
}

TEST(resCheck, check_11) {
  int expectedVals[] = {1};

  auto res = balls_and_bins(1, 1);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(1, 1, expectedVals)), 1);
}

TEST(resCheck, check_12) {
  int expectedVals[] = {1, 0,
                        0, 1};

  auto res = balls_and_bins(1, 2);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(2, 2, expectedVals)), 1);
}

TEST(resCheck, check_21) {
  int expectedVals[] = {2};

  auto res = balls_and_bins(2, 1);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(1, 1, expectedVals)), 1);
}

TEST(resCheck, check_22) {
  int expectedVals[] = {2, 0,
                        1, 1,
                        0, 2};

  auto res = balls_and_bins(2, 2);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(3, 2, expectedVals)), 1);
}

TEST(resCheck, check_31) {
  int expectedVals[] = {3};

  auto res = balls_and_bins(3, 1);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(1, 1, expectedVals)), 1);
}

TEST(resCheck, check_32) {
  int expectedVals[] = {3, 0,
                        2, 1,
                        1, 2,
                        0, 3};

  auto res = balls_and_bins(3, 2);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(4, 2, expectedVals)), 1);
}

TEST(resCheck, check_33) {
  int expectedVals[] = {3, 0, 0,
                        2, 1, 0,
                        2, 0, 1,
                        1, 2, 0,
                        1, 1, 1,
                        1, 0, 2,
                        0, 3, 0,
                        0, 2, 1,
                        0, 1, 2,
                        0, 0, 3};

  auto res = balls_and_bins(3, 3);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(isIntMatrixEqual(res[0], cv::Mat1i(10, 3, expectedVals)), 1);
}