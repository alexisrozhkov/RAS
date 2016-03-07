// Copyright 2016 Alexey Rozhkov

#ifndef CORE_UTILS_SUBSPACE_ANGLE_H_
#define CORE_UTILS_SUBSPACE_ANGLE_H_

#include <core/utils/mat_nd.h>


//  SUBSPACE_GPCA
//     Angle between subspaces.
//     SUBSPACE_ANGLE(A,B) finds the angle between two
//     subspaces specified by the columns of A and B.
//
//  Same as MATLAB's built in SUBSPACE command, but assumes that A and B are
//  already orthonormal, saving a couple extra SVD's and speeding up the
//  code.
EmbValT subspace_angle(const Mat2D &A, const Mat2D &B);

#endif  // CORE_UTILS_SUBSPACE_ANGLE_H_
