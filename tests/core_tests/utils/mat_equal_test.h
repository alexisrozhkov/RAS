// Copyright 2016 Alexey Rozhkov

#ifndef TESTS_CORE_TESTS_UTILS_MAT_EQUAL_TEST_H_
#define TESTS_CORE_TESTS_UTILS_MAT_EQUAL_TEST_H_

#include <core/utils/mat_nd.h>
#include <gtest/gtest.h>


testing::AssertionResult isIntMatrixEqual(const IndexMat2D &a,
                                          const IndexMat2D &b);

testing::AssertionResult isDblMatrixEqual(const Mat2D &a,
                                          const Mat2D &b,
                                          const EmbValT epsilon = 1e-10);

#endif  // TESTS_CORE_TESTS_UTILS_MAT_EQUAL_TEST_H_
