// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <core/find_polynomials.h>


Mat2D robust_algebraic_segmentation(const Mat2D &img1,
                                    const Mat2D &img2,
                                    const int groupCount,

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

  return Mat2D();
}
