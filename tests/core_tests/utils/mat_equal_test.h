//
// Created by alexey on 27.02.16.
//

#ifndef RAS_MAT_EQUAL_TEST_H
#define RAS_MAT_EQUAL_TEST_H

#include <gtest/gtest.h>
#include <ras_types.h>


testing::AssertionResult isIntMatrixEqual(const IndexMat2D &a, const IndexMat2D &b);
testing::AssertionResult isDblMatrixEqual(const Mat2D &a, const Mat2D &b, const EmbValT epsilon=1e-10);

#endif //RAS_MAT_EQUAL_TEST_H
