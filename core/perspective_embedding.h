//
// Created by alexey on 22.02.16.
//

#ifndef RAS_PERSPECTIVE_EMBEDDING_H
#define RAS_PERSPECTIVE_EMBEDDING_H

#include <tuple>
#include <vector>
#include <opencv2/core.hpp>

typedef cv::Mat1d Mat2D;
typedef std::vector<Mat2D> Mat2DArray;

typedef Mat2DArray Mat3D;
typedef std::vector<Mat3D> Mat3DArray;

typedef Mat3DArray Mat4D;
typedef std::vector<Mat4D> Mat4DArray;

std::tuple<Mat2DArray, Mat3DArray, Mat4DArray> perspective_embedding(const Mat2D& data,
                                                        const unsigned int order,
                                                        const bool all=false,
                                                        const int nargout=3);

#endif //RAS_PERSPECTIVE_EMBEDDING_H
