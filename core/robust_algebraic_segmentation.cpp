// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <core/perspective_embedding.h>
#include <core/utils/cholesky.h>
#include <core/utils/subspace_angle.h>
#include <iostream>


Mat2D robust_algebraic_segmentation(const Mat2D &img1,
                                    const Mat2D &img2,
                                    const unsigned int groupCount,

                                    const int debug,
                                    const bool postRansac,
                                    const EmbValT angleTolerance,
                                    const EmbValT boundaryThreshold,
                                    const FindPolyMethod fittingMethod,
                                    const InfluenceMethod influenceMethod,
                                    const bool normalizeCoordinates,
                                    const bool normalizeQuadratics,
                                    const EmbValT minOutlierPercentage,
                                    const EmbValT maxOutlierPercentage,
                                    const bool retestOutliers) {
  //////////////////////////////////////////////////////////////////////////////
  // parse arguments (currently as close as possible to Matlab, should be
  // refactored later
  const int DEBUG = debug;
  const bool POST_RANSAC = postRansac;

  CV_Assert(!(angleTolerance < 0 || angleTolerance >= CV_PI/2));
  CV_Assert(!(boundaryThreshold < 0));

  // should be false if boundaryThreshold is not specified
  const bool REJECT_UNKNOWN_OUTLIERS = true;

  const FindPolyMethod FITTING_METHOD = fittingMethod;
  const InfluenceMethod INFLUENCE_METHOD = influenceMethod;

  // todo: check if it is an error/typo in Matlab implementation
  // const bool NORMALIZE_COORDINATES = true;
  const bool NORMALIZE_DATA = normalizeCoordinates;

  const bool NORMALIZE_QUADRATICS = normalizeQuadratics;

  CV_Assert(!(maxOutlierPercentage < 0 || maxOutlierPercentage >= 1));
  CV_Assert(!(minOutlierPercentage < 0 || minOutlierPercentage >= 1));
  CV_Assert(!(minOutlierPercentage > maxOutlierPercentage));

  // should be false if min-/maxOutlierPercentage is not specified
  const bool REJECT_KNOWN_OUTLIERS = true;

  const bool RETEST_OUTLIERS = retestOutliers;

  //////////////////////////////////////////////////////////////////////////////
  // Step 1: map coordinates to joint image space.
  const int sampleCount = img1.cols;
  const int dimensionCount = 5;
  Mat2D jointImageData = Mat2D::zeros(dimensionCount, sampleCount);

  if (NORMALIZE_DATA) {
    Mat2D img1_, img2_, temp;
    cv::vconcat(img1, Mat2D::ones(1, sampleCount), img1_);
    cv::vconcat(img2, Mat2D::ones(1, sampleCount), img2_);

    temp = Cholesky((img1_*img1_.t()/sampleCount).inv())*img1_;
    temp.rowRange(0, 2).copyTo(jointImageData.rowRange(0, 2));

    temp = Cholesky((img2_*img2_.t()/sampleCount).inv())*img2_;
    temp.rowRange(0, 2).copyTo(jointImageData.rowRange(2, 4));
  } else {
    img1.copyTo(jointImageData.rowRange(0, 2));
    img2.copyTo(jointImageData.rowRange(2, 4));
  }

  jointImageData.row(4) = Mat2D::ones(1, sampleCount);

  //////////////////////////////////////////////////////////////////////////////
  // Step 2: apply perspective veronese map to data
  auto embedding = perspective_embedding(jointImageData, groupCount, false);
  auto veroneseData = embedding.getV().back();
  auto veroneseDerivative = embedding.getD().back();
  auto veroneseHessian = embedding.getH().back();

  //////////////////////////////////////////////////////////////////////////////
  // Step 3: Use influence function to perform robust pca

  if (REJECT_KNOWN_OUTLIERS || REJECT_UNKNOWN_OUTLIERS) {
    if (INFLUENCE_METHOD == InfluenceMethod::SAMPLE) {
      auto polynomialCoefficients = find_polynomials(veroneseData,
                                                     veroneseDerivative,
                                                     FITTING_METHOD,
                                                     1);

      // Reject outliers by the sample influence function
      Mat2D influenceValues = Mat2D::zeros(1, sampleCount);

      for (int sIdx = 0; sIdx < sampleCount; sIdx++) {
        // compute the leave-one-out influence
        auto U = find_polynomials(veroneseData,
                                  veroneseDerivative,
                                  FITTING_METHOD,
                                  1,
                                  sIdx);

        influenceValues(sIdx) = subspace_angle(polynomialCoefficients, U);
      }

      std::cout << influenceValues << std::endl;
    } else {
      // todo
    }
  }

  return veroneseData;
}
