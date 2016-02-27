//
// Created by alexey on 27.02.16.
//

#ifndef RAS_MAT_EQUAL_TEST_H
#define RAS_MAT_EQUAL_TEST_H

#include <gtest/gtest.h>
#include <perspective_embedding.h>

testing::AssertionResult isIntMatrixEqual(const cv::Mat1i& a, const cv::Mat1i& b);
testing::AssertionResult isDblMatrixEqual(const Mat2D& a, const Mat2D& b);

#endif //RAS_MAT_EQUAL_TEST_H
