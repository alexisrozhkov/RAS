//
// Created by alexey on 28.02.16.
//

#include <Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>

#include "generalized_eigenvalues.h"


void generalizedEigenvals(const Mat2D &A, const Mat2D &B, Mat2D &al, Mat2D &be) {
  typedef Eigen::Matrix<EmbValT, Eigen::Dynamic, Eigen::Dynamic> EigMat2D;

  EigMat2D A_, B_;

  cv::cv2eigen(A, A_);
  cv::cv2eigen(B, B_);

  const Eigen::RealQZ<EigMat2D> rqz(A_, B_);

  cv::eigen2cv(EigMat2D(rqz.matrixS().diagonal()), al);
  cv::eigen2cv(EigMat2D(rqz.matrixT().diagonal()), be);
}