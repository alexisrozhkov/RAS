//
// Created by alexey on 25.02.2016.
//

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
       if(fabs(a(j, i) - b(j, i)) > comparisonEpsilon) {
          return testing::AssertionFailure() << testing::Message("Matrix content mismatch: ")
               << a(j, i) << testing::Message(" != ") <<  b(j, i);
       }
     }
  }

  return testing::AssertionSuccess();
}

Embedding getExpectedEmbedding(const int K, const int N) {
  assert(K >= 4 && K <= 5);
  assert(N >= 1 && N <= 2);

  const std::vector<double> *lookup = expectedVals[K-K_offset][N-N_offset];
  int dims = lookup[0].size()/N;

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

  for(int k = 0; k < dims; k++) {
    H[k] = Mat3D((uint)K);

    for(int k_ = 0; k_ < K; k_++) {
      H[k][k_] = Mat2D(K, N);

      for(int j = 0; j < K; j++) {
        for(int i = 0; i < N; i++) {
          H[k][k_](j, i) = lookup[2][((i*K + j)*dims + k)*K + k_];
        }
      }
    }
  }

  return std::make_tuple(Mat2DArray{V},
                         Mat3DArray{D},
                         Mat4DArray{H});
}

class PerspectiveEmbeddingTest : public testing::TestWithParam<std::tuple<int, int>>
{
 public:
  virtual void SetUp(){}
  virtual void TearDown(){}
};

TEST(PerspectiveEmbeddingTest, checkDimensionValidation) {
  //todo: replace assert with exception inside
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(0, 1), 1), "");
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(1, 1), 1), "");
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(2, 1), 1), "");
  EXPECT_DEATH(perspective_embedding(Mat2D::ones(3, 1), 1), "");
  perspective_embedding(Mat2D::ones(4, 1), 1);
  perspective_embedding(Mat2D::ones(5, 1), 1);
}

void checkEmbeddingEqual(const Embedding &a, const Embedding &b) {
  auto V = std::get<0>(a).back();
  auto D = std::get<1>(a).back();
  auto H = std::get<2>(a).back();

  auto Ve = std::get<0>(b).back();
  auto De = std::get<1>(b).back();
  auto He = std::get<2>(b).back();

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
                        ::testing::Combine(testing::Range(0, 2),
                                           testing::Range(0, 2)));

TEST_P(PerspectiveEmbeddingTest, checkResult) {
  const int Kidx = std::get<0>(GetParam()),
            Nidx = std::get<1>(GetParam()),
            O = 1;

  const int K = Kidx+K_offset, N = Nidx+N_offset;
  Mat2D input = Mat2D(inputData[Kidx][Nidx]).reshape(0, K);

  checkEmbeddingEqual(perspective_embedding(input, O),
                      getExpectedEmbedding(K, N));
}