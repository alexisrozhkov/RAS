//
// Created by alexey on 25.02.2016.
//

#include <cmath>
#include <tuple>
#include <gtest/gtest.h>
#include <mat_equal_test.h>
#include <perspective_embedding.h>

#include "perspective_embedding_data.h"



class PerspectiveEmbeddingTest : public testing::TestWithParam<std::tuple<bool, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};

// this could have been implemented as a comparison operator of Embedding, but this way it provides a more verbose way
// of testing equality (dimension and data mismatches are reported separately, besides mismatching values are shown)
void checkEmbeddingEqual(const Embedding &a, const Embedding &b) {
  const auto V = a.getV().back();
  const auto D = a.getD().back();
  const auto H = a.getH().back();

  const auto Ve = b.getV().back();
  const auto De = b.getD().back();
  const auto He = b.getH().back();

  EXPECT_TRUE(isDblMatrixEqual(V, Ve));

  EXPECT_EQ(D.size(), De.size());
  for(int i = 0; i < D.size(); i++) {
    EXPECT_TRUE(isDblMatrixEqual(D[i], De[i]));
  }

  EXPECT_EQ(H.size(), He.size());
  for(int i = 0; i < H.size(); i++) {
    EXPECT_EQ(H[0].size(), He[0].size());
    for(int j = 0; j < H[0].size(); j++) {
      EXPECT_TRUE(isDblMatrixEqual(H[i][j], He[i][j]));
    }
  }
}

INSTANTIATE_TEST_CASE_P(First22Arguments,
                        PerspectiveEmbeddingTest,
                        ::testing::Combine(testing::Bool(),
                                           testing::Range(1, 3)));

TEST_P(PerspectiveEmbeddingTest, check1Motion) {
  const bool hasZeros = std::get<0>(GetParam());
  const int N = std::get<1>(GetParam()), O = 1;

  assert(N >= 1 && N <= 2);

  const int Nidx = N-N_offset;

  const Mat2D input = Mat2D(oneMotionInput[hasZeros][Nidx]).reshape(0, Kconst);

  checkEmbeddingEqual(perspective_embedding(input, O),
                      Embedding(oneMotionExpected[hasZeros][Nidx], N));
}

TEST(PerspectiveEmbeddingTest, check2Motions) {
  const int N = 2, O = 2;

  const Mat2D input = Mat2D(twoMotionsInput).reshape(0, Kconst);

  checkEmbeddingEqual(perspective_embedding(input, O),
                      Embedding(twoMotionsExpected, N));
}

TEST(PerspectiveEmbeddingTest, checkDimensionValidation) {
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(Kconst-1, 1), 1), "");
  perspective_embedding(Mat2D::ones(Kconst, 1), 1);
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(Kconst+1, 1), 1), "");
}