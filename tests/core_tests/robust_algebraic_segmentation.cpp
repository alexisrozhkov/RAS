// Copyright 2016 Alexey Rozhkov

#include <tests/core_tests/robust_algebraic_segmentation_data.h>
#include <tests/core_tests/utils/mat_equal_test.h>
#include <core/robust_algebraic_segmentation.h>
#include <core/utils/mat_nd.h>
#include <gtest/gtest.h>


class RobustAlgebraicSegmentationTest : public testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

void checkSegmentationEqual(const SegmentationResult &a,
                            const SegmentationResult &b) {
  EXPECT_TRUE(isIntMatrixEqual(a.labels, b.labels));
  EXPECT_TRUE(isIntMatrixEqual(a.motionModels, b.motionModels));
  EXPECT_TRUE(isIntMatrixEqual(a.allLabels, b.allLabels));

  for (size_t i = 0; i < a.quadratics.size(); i++) {
    EXPECT_TRUE(isDblMatrixEqual(a.quadratics[i], b.quadratics[i], 1e-5));
  }
}

TEST(RobustAlgebraicSegmentationTest, compareResultWithOctave) {
  auto segm = robust_algebraic_segmentation(Mat2D(img1, true).reshape(0, 2),
                                            Mat2D(img2, true).reshape(0, 2),
                                            2,
                                            RAS_params(0.25, {3*CV_PI/180}));

  SegmentationResult expected {
      IndexMat2D(labels, true).reshape(0, 1)-1,
      IndexMat2D(motionModels, true).reshape(0, 1),
      Mat3D{Mat2D(quadratics[0], true).reshape(0, 5),
            Mat2D(quadratics[1], true).reshape(0, 5)},
      IndexMat2D(labelsAll, true).reshape(0, 1)-1
  };

  checkSegmentationEqual(segm, expected);
}
