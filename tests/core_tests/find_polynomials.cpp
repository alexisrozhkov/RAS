//
// Created by alexey on 27.02.16.
//

#include <gtest/gtest.h>
#include <find_polynomials.h>

class FindPolynomialsTest : public testing::TestWithParam<std::tuple<int, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};


TEST(FindPolynomialsTest, dummy) {
  ASSERT_EQ(1, 1);
}