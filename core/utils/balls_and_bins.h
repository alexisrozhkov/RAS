//
// Created by alexey on 21.02.16.
//

#ifndef RAS_BALLS_AND_BINS_H
#define RAS_BALLS_AND_BINS_H

#include <vector>
#include <opencv2/core.hpp>

// Enumerates all possible groupings of identical objects, i.e. balls in bins.
// Note: This happens to be the same problem as computing the exponents of the veronese map.
std::vector<cv::Mat1i> balls_and_bins(const unsigned int ballCount,
                                      const unsigned int binCount,
                                      const bool all=0);

#endif //RAS_BALLS_AND_BINS_H
