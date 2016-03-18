// Copyright 2016 Alexey Rozhkov

#include <core/utils/ras_params.h>
#include <vector>


RAS_params::RAS_params(const EmbValT bT,
                       const std::vector<EmbValT> &angTol,
                       const EmbValT miOP,
                       const EmbValT maOP,
                       const int debug,
                       const bool postRansac,
                       const FindPolyMethod fittingMethod,
                       const InfluenceMethod influenceMethod,
                       const bool normalizeCoordinates,
                       const bool normalizeQuadratics,
                       const bool retestOutliers) :
    DEBUG(debug),
    POST_RANSAC(postRansac),
    boundaryThreshold((bT == NotSpecified) ? 0 : bT),

    FITTING_METHOD(fittingMethod),
    INFLUENCE_METHOD(influenceMethod),

    NORMALIZE_DATA(normalizeCoordinates),
    NORMALIZE_QUADRATICS(normalizeQuadratics),

    minOutlierPercentage(chooseMiOP(miOP, maOP)),
    maxOutlierPercentage(chooseMaOP(miOP, maOP)),

    REJECT_UNKNOWN_OUTLIERS(chooseRUO(bT, miOP, maOP)),
    REJECT_KNOWN_OUTLIERS(chooseRKO(miOP, maOP)),

    RETEST_OUTLIERS(retestOutliers),
    angleTolerance(angTol) {
  for (size_t i = 0; i < angleTolerance.size(); i++) {
    CV_Assert(!(angleTolerance[i] < 0 || angleTolerance[i] >= CV_PI / 2));
  }

  // todo: handle unspecified value of boundaryThreshold more gracefully

  // will not complain if boundaryThreshold_ was -1, since it's a default value
  // and will be overwritten with 0
  CV_Assert(boundaryThreshold >= 0);

  // todo: check if it is an error/typo in Matlab implementation
  // const bool NORMALIZE_COORDINATES = true;
  // const bool NORMALIZE_DATA = normalizeCoordinates;

  CV_Assert(!(maxOutlierPercentage < 0 || maxOutlierPercentage >= 1));
  CV_Assert(!(minOutlierPercentage < 0 || minOutlierPercentage >= 1));
  CV_Assert(minOutlierPercentage <= maxOutlierPercentage);
}

EmbValT RAS_params::chooseMiOP(const EmbValT miOP, const EmbValT maOP) {
  const EmbValT defaultVal = 0;

  if (miOP == NotSpecified) {  // nothing given
    return defaultVal;
  } else {  // one or two values given
    return miOP;
  }
}

EmbValT RAS_params::chooseMaOP(const EmbValT miOP, const EmbValT maOP) {
  const EmbValT defaultVal = 0.5;

  if (miOP == NotSpecified) {  // nothing given
    return defaultVal;
  } else if (maOP == NotSpecified) {  // one value given
    return miOP;
  } else {
    return maOP;  // range given
  }
}

bool RAS_params::chooseRKO(const EmbValT miOP, const EmbValT maOP) {
  return (miOP > 0) && (maOP == NotSpecified);
}

bool RAS_params::chooseRUO(const EmbValT boundaryThreshold,
                           const EmbValT miOP,
                           const EmbValT maOP) {
  return boundaryThreshold != NotSpecified ||
      ((miOP != NotSpecified) && (maOP != NotSpecified));
}
