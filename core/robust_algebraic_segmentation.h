// Copyright 2016 Alexey Rozhkov

#ifndef CORE_ROBUST_ALGEBRAIC_SEGMENTATION_H_
#define CORE_ROBUST_ALGEBRAIC_SEGMENTATION_H_

#include <core/utils/ras_types.h>
#include <core/utils/ras_params.h>
#include <core/find_polynomials.h>
#include <vector>

struct SegmentationResult {
  IndexMat2D labels;
  IndexMat2D motionModels;
  Mat3D quadratics;
  IndexMat2D allLabels;
};

const int outlierIdx = -1;

// Segment noisy and outlier-ridden feature correspondences in a
// 2-view perspective dynamic scene using Robust Algebraic Segmentation
SegmentationResult robust_algebraic_segmentation(const Mat2D &img1,
                                                 const Mat2D &img2,
                                                 const unsigned int groupCount,
                                                 const RAS_params &params);

#endif  // CORE_ROBUST_ALGEBRAIC_SEGMENTATION_H_
