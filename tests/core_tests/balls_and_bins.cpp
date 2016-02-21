//
// Created by alexey on 21.02.16.
//

#include "gtest/gtest.h"
#include <balls_and_bins.h>


TEST(bar_test, check_0) {
  EXPECT_NE(bar(), 1);
}

TEST(bar_test, check_1) {
  EXPECT_EQ(bar(), 2);
}

TEST(bar_test, check_2) {
  EXPECT_NE(bar(), 3);
}