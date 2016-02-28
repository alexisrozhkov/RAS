//
// Created by alexey on 28.02.16.
//

#ifndef RAS_RAS_TYPES_H
#define RAS_RAS_TYPES_H

#include <array>
#include <vector>
#include <ostream>
#include <opencv2/core.hpp>


typedef int IndValT;
typedef double EmbValT;


typedef cv::Mat_<IndValT> IndexMat2D;
typedef std::vector<IndexMat2D> IndexMat2DArray;
typedef std::vector<IndexMat2DArray> IndexMat2DArray2D;


typedef cv::Mat_<EmbValT> Mat2D;
typedef std::vector<Mat2D> Mat2DArray;

typedef Mat2DArray Mat3D;
typedef std::vector<Mat3D> Mat3DArray;

typedef Mat3DArray Mat4D;
typedef std::vector<Mat4D> Mat4DArray;


typedef std::array<std::vector<EmbValT>, 3> EmbeddingInitializer;


std::ostream &operator<<(std::ostream &os, Mat3D const &m);
std::ostream &operator<<(std::ostream &os, Mat4D const &m);

#endif //RAS_RAS_TYPES_H
