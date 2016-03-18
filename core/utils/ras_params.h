// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_RAS_PARAMS_H_
#define CORE_UTILS_RAS_PARAMS_H_

#include <core/utils/ras_types.h>
#include <core/find_polynomials.h>
#include <vector>


enum class InfluenceMethod {
  SAMPLE,
  THEORETICAL
};

const EmbValT NotSpecified = -1;


class RAS_params {
  EmbValT chooseMiOP(const EmbValT miOP,
                     const EmbValT maOP);

  EmbValT chooseMaOP(const EmbValT miOP,
                     const EmbValT maOP);

  bool chooseRKO(const EmbValT miOP,
                 const EmbValT maOP);

  bool chooseRUO(const EmbValT boundaryThreshold,
                 const EmbValT miOP,
                 const EmbValT maOP);

 public:
  const int DEBUG;
  const bool POST_RANSAC;
  const EmbValT boundaryThreshold;

  const FindPolyMethod FITTING_METHOD;
  const InfluenceMethod INFLUENCE_METHOD;

  const bool NORMALIZE_DATA;
  const bool NORMALIZE_QUADRATICS;

  const EmbValT minOutlierPercentage;
  const EmbValT maxOutlierPercentage;

  const bool REJECT_UNKNOWN_OUTLIERS;
  const bool REJECT_KNOWN_OUTLIERS;

  const bool RETEST_OUTLIERS;
  const std::vector<EmbValT> angleTolerance;

  RAS_params(const EmbValT boundaryThreshold = NotSpecified,
             const std::vector<EmbValT> &angleTolerance = {CV_PI/60},
             const EmbValT minOutlierPercentage = NotSpecified,
             const EmbValT maxOutlierPercentage = NotSpecified,
             const int debug = 0,
             const bool postRansac = true,
             const FindPolyMethod fittingMethod = FindPolyMethod::FISHER,
             const InfluenceMethod influenceMethod = InfluenceMethod::SAMPLE,
             const bool normalizeCoordinates = true,
             const bool normalizeQuadratics = true,
             const bool retestOutliers = false);
};

#endif  // CORE_UTILS_RAS_PARAMS_H_
