// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_BALLS_AND_BINS_H_
#define CORE_UTILS_BALLS_AND_BINS_H_

#include <core/utils/ras_types.h>


// Enumerates all possible groupings of identical objects, i.e. balls in bins.
// Note: This happens to be the same problem as computing the exponents of the
// veronese map.
IndexMat2DArray balls_and_bins(const unsigned int ballCount,
                               const unsigned int binCount,
                               const bool all = 0);

#endif  // CORE_UTILS_BALLS_AND_BINS_H_
