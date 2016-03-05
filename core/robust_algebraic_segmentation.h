// Copyright 2016 Alexey Rozhkov

#ifndef CORE_ROBUST_ALGEBRAIC_SEGMENTATION_H_
#define CORE_ROBUST_ALGEBRAIC_SEGMENTATION_H_

#include <core/utils/ras_types.h>
#include <core/find_polynomials.h>


enum class InfluenceMethod {
  SAMPLE,
  THEORETICAL
};

// Segment noisy and outlier-ridden feature correspondences in a
// 2-view perspective dynamic scene using Robust Algebraic Segmentation
Mat2D robust_algebraic_segmentation(const Mat2D &img1,
                                    const Mat2D &img2,
                                    const unsigned int groupCount,

                                    const int debug = 0,
                                    const bool postRansac = true,
                                    const EmbValT angleTolerance = CV_PI/60,
                                    const EmbValT boundaryThreshold = 0,
                                    const FindPolyMethod fittingMethod =
                                          FindPolyMethod::FISHER,
                                    const InfluenceMethod influenceMethod =
                                          InfluenceMethod::SAMPLE,
                                    const bool normalizeCoordinates = true,
                                    const bool normalizeQuadratics = true,
                                    const EmbValT minOutlierPercentage = 0.0,
                                    const EmbValT maxOutlierPercentage = 0.5,
                                    const bool retestOutliers = false);

#endif  // CORE_ROBUST_ALGEBRAIC_SEGMENTATION_H_
