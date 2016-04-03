// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_MAT_ND_H_
#define CORE_UTILS_MAT_ND_H_

#include <core/utils/ras_types.h>
#include <vector>


typedef cv::Mat_<EmbValT> Mat2D;
typedef std::vector<Mat2D> Mat2DArray;

typedef Mat2DArray Mat3D;
typedef std::vector<Mat3D> Mat3DArray;

typedef Mat3DArray Mat4D;
typedef std::vector<Mat4D> Mat4DArray;


Mat3D Mat3D_zeros(const int A, const int B, const int C);
Mat4D Mat4D_zeros(const int A, const int B, const int C, const int D);

Mat2D Mat2D_clone(const Mat2D &from);
Mat3D Mat3D_clone(const Mat3D &from);
Mat4D Mat4D_clone(const Mat4D &from);

Mat2D filterIdx2(const Mat2D &src, const std::vector<int> &indices);
Mat3D filterIdx3(const Mat3D &src, const std::vector<int> &indices);
Mat4D filterIdx4(const Mat4D &src, const std::vector<int> &indices);

Mat2D sliceIdx2(const Mat3D &src, const int idx);
Mat2D sliceIdx3(const Mat3D &src, const int idx);
Mat2D sliceIdx23(const Mat4D &src, const int idx1, const int idx2);

Mat2D meanIdx3(const Mat3D &src);

std::ostream &operator<<(std::ostream &os, Mat3D const &m);
std::ostream &operator<<(std::ostream &os, Mat4D const &m);

#endif  // CORE_UTILS_MAT_ND_H_
