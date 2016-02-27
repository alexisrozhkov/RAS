//
// Created by alexey on 25.02.2016.
//

#include <cmath>
#include <tuple>

#include "gtest/gtest.h"
#include <perspective_embedding.h>
#include "perspective_embedding_data.h"


const double comparisonEpsilon = 1e-10;

testing::AssertionResult isDblMatrixEqual(const Mat2D& a, const Mat2D& b) {
  if(a.type() != b.type()) {
     return testing::AssertionFailure() << testing::Message("Matrix type mismatch");
  }

  if((a.cols != b.cols) || (a.rows != b.rows)) {
     return testing::AssertionFailure() << testing::Message("Matrix size mismatch");
  }

  for(int j = 0; j < a.rows; j++) {
     for(int i = 0; i < a.cols; i++) {
       if((std::isnan(a(j, i))) || (std::isnan(b(j, i))) || (fabs(a(j, i) - b(j, i)) > comparisonEpsilon)) {
          return testing::AssertionFailure() << testing::Message("Matrix content mismatch: ")
               << a(j, i) << testing::Message(" != ") <<  b(j, i);
       }
     }
  }

  return testing::AssertionSuccess();
}

Embedding getExpectedEmbedding(const std::vector<double> *lookup, const int N) {
  const int dims = int(lookup[0].size())/N, K = Kconst;

  Mat2D V = Mat2D(lookup[0]).reshape(0, dims);

  Mat3D D((uint)dims);
  for(int k = 0; k < dims; k++) {
    D[k] = Mat2D(K, N);

    for(int j = 0; j < K; j++) {
      for(int i = 0; i < N; i++) {
        D[k](j, i) = lookup[1][(i*dims + k)*K + j];
      }
    }
  }

  Mat4D H((uint)dims);

  for(int l = 0; l < dims; l++) {
    H[l] = Mat3D((uint)K);

    for(int k = 0; k < K; k++) {
      H[l][k] = Mat2D(K, N);

      for(int j = 0; j < K; j++) {
        for(int i = 0; i < N; i++) {
          H[l][k](j, i) = lookup[2][((i*K + j)*dims + l)*K + k];
        }
      }
    }
  }

  return Embedding(Mat2DArray{V},
                   Mat3DArray{D},
                   Mat4DArray{H});
}

Embedding getExpectedEmbeddingFromArr(const bool zeros, const int N) {
  assert(N >= 1 && N <= 2);
  const std::vector<double> *lookup = expectedVals[zeros][N-N_offset];
  return getExpectedEmbedding(lookup, N);
}

class PerspectiveEmbeddingTest : public testing::TestWithParam<std::tuple<bool, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};

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

  const Mat2D input = Mat2D(inputData[hasZeros][N-N_offset]).reshape(0, Kconst);

  checkEmbeddingEqual(perspective_embedding(input, O),
                      getExpectedEmbeddingFromArr(hasZeros, N));
}

TEST(PerspectiveEmbeddingTest, check2Motions) {
  const int N = 2, O = 2;

  const Mat2D input = Mat2D(twoMotionsInput).reshape(0, Kconst);

  checkEmbeddingEqual(perspective_embedding(input, O),
                      getExpectedEmbedding(twoMotionsExpected, N));
}

TEST(PerspectiveEmbeddingTest, checkDimensionValidation) {
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(Kconst-1, 1), 1), "");
  perspective_embedding(Mat2D::ones(Kconst, 1), 1);
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(Kconst+1, 1), 1), "");
}