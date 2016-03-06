// Copyright 2016 Alexey Rozhkov

#include <3rdparty/Eigen/SVD>
#include <opencv2/core/eigen.hpp>
#include <core/utils/jacobi_svd.h>


void jacobiSVD_U(const Mat2D &A, Mat2D *U) {
  typedef Eigen::Matrix<EmbValT, Eigen::Dynamic, Eigen::Dynamic> EigMat2D;

  EigMat2D A_;
  cv::cv2eigen(A, A_);

  // HouseholderQRPreconditioner gives result closest to Matlab, but may be
  // not the most accurate according to Eigen's doc
  Eigen::JacobiSVD<EigMat2D, Eigen::HouseholderQRPreconditioner>
      svd(A_, Eigen::ComputeFullU);

  cv::eigen2cv(EigMat2D(svd.matrixU()), *U);
}




