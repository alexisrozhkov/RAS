//
// Created by alexey on 22.02.16.
//

#ifndef RAS_PERSPECTIVE_EMBEDDING_H
#define RAS_PERSPECTIVE_EMBEDDING_H

#include <tuple>
#include <vector>
#include <opencv2/core.hpp>

typedef std::vector<cv::Mat> Mat2DArray;
typedef Mat2DArray Mat3D;
typedef std::vector<Mat3D> Mat3DArray;
typedef Mat3DArray Mat4D;
typedef std::vector<Mat4D> Mat4DArray;
typedef Mat4DArray Mat5D;

std::tuple<Mat2DArray, Mat3DArray, Mat4DArray> perspective_embedding(const cv::Mat& data,
                                                        const unsigned int order,
                                                        const bool all,
                                                        const int nargout);

#endif //RAS_PERSPECTIVE_EMBEDDING_H
