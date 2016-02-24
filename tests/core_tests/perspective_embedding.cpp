//
// Created by alexey on 25.02.2016.
//

#include "gtest/gtest.h"
#include <perspective_embedding.h>

const double comparisonEpsilon = 1e-10;

const std::vector<double> expectedVals41[3] = {
  {3.0000,
   4.0000,
   6.0000,
   8.0000},

  {3.00000,   0.00000,   1.00000,   0.00000,
   4.00000,   0.00000,   0.00000,   1.00000,
   0.00000,   3.00000,   2.00000,   0.00000,
   0.00000,   4.00000,   0.00000,   2.00000},

  {0,   0,   1,   0,
   0,   0,   0,   1,
   0,   0,   0,   0,
   0,   0,   0,   0,

   0,   0,   0,   0,
   0,   0,   0,   0,
   0,   0,   1,   0,
   0,   0,   0,   1,

   1,   0,   0,   0,
   0,   0,   0,   0,
   0,   1,   0,   0,
   0,   0,   0,   0,

   0,   0,   0,   0,
   1,   0,   0,   0,
   0,   0,   0,   0,
   0,   1,   0,   0}
};

std::tuple<Mat2D, Mat3D, Mat4D> getExpectedMat(const std::vector<double>* expectedValsA) {
  Mat2D V = Mat2D(expectedValsA[0]).t();
  V = V.reshape(0, 4);

  Mat3D D(4);
  for(int k = 0; k < D.size(); k++) {
    D[k] = Mat2D(4, 1);

    for(int j = 0; j < D[0].rows; j++) {
      for(int i = 0; i < D[0].cols; i++) {
        D[k](j, i) = expectedValsA[1][i*(D.size()*D[0].rows) + k*D[0].rows + j];
      }
    }
  }

  Mat4D H(4);

  for(int k = 0; k < H.size(); k++) {
    H[k] = Mat3D(4);

    for(int k_ = 0; k_ < H[0].size(); k_++) {
      H[k][k_] = Mat2D(4, 1);

      for(int j = 0; j < H[0][0].rows; j++) {
        for(int i = 0; i < H[0][0].cols; i++) {
          H[k][k_](j, i) = expectedValsA[2][i*(H[0].size()*H.size()*H[0][0].rows) +
                                            j*(H[0].size()*H.size()) +
                                            k*H[0].size() +
                                            k_];
        }
      }
    }
  }

  return std::make_tuple(V, D, H);
}

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

TEST(PerspectiveEmbeddingTest, checkResult) {
  double matchesData[] = {1,
                          2,
                          3,
                          4};

  auto tpl = perspective_embedding(Mat2D(4, 1, matchesData), 1);
  auto tpl2 = getExpectedMat(expectedVals41);

  auto V = std::get<0>(tpl).back();
  auto D = std::get<1>(tpl).back();
  auto H = std::get<2>(tpl).back();

  auto Ve = std::get<0>(tpl2);
  auto De = std::get<1>(tpl2);
  auto He = std::get<2>(tpl2);

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