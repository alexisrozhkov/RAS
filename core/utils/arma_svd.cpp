// Copyright 2016 Alexey Rozhkov

#include <core/utils/arma_svd.h>
#include <armadillo>


void armaSVD_U(const Mat2D &A, Mat2D *U_) {
  typedef arma::Col<EmbValT> AMat1D;
  typedef arma::Mat<EmbValT> AMat2D;

  // it is important that the data pointer is taken from transposed data,
  // and the dimensions are swapped, since Armadillo stores data in column-major
  // format, unlike OpenCV
  Mat2D At = A.t();
  AMat2D armaMat(reinterpret_cast<EmbValT*>(At.data), At.cols, At.rows);

  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, armaMat, "std");

  // transpose is needed here not only to switch back to row-major, but also
  // to copy the data, because when cv::Mat is constructed from provided data
  // it doesn't copy it, and it seems that there is no flag to force copying
  *U_ = Mat2D((int)U.n_cols, (int)U.n_rows, U.memptr()).t();
}
