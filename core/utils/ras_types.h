// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_RAS_TYPES_H_
#define CORE_UTILS_RAS_TYPES_H_

#include <opencv2/core.hpp>

#include <array>
#include <vector>
#include <ostream>


typedef int IndValT;
typedef double EmbValT;


typedef cv::Mat_<IndValT> IndexMat2D;
typedef std::vector<IndexMat2D> IndexMat2DArray;
typedef std::vector<IndexMat2DArray> IndexMat2DArray2D;

typedef std::array<std::vector<EmbValT>, 3> EmbeddingInitializer;

#endif  // CORE_UTILS_RAS_TYPES_H_
