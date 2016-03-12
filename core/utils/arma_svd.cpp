// Copyright 2016 Alexey Rozhkov

#include <core/utils/arma_svd.h>
#include <armadillo>

typedef arma::Col<EmbValT> AMat1D;
typedef arma::Mat<EmbValT> AMat2D;


AMat2D cv2arma(const Mat2D &src) {
  // it is important that the data pointer is taken from transposed data,
  // and the dimensions are swapped, since Armadillo stores data in column-major
  // format, unlike OpenCV
  Mat2D At = src.t();
  return AMat2D(reinterpret_cast<EmbValT*>(At.data), At.cols, At.rows);
}

Mat2D arma2cv(AMat2D &src) {
  // transpose is needed here not only to switch back to row-major, but also
  // to copy the data, because when cv::Mat is constructed from provided data
  // it doesn't copy it, and it seems that there is no flag to force copying
  return Mat2D((int)src.n_cols, (int)src.n_rows, src.memptr()).t();
}

void armaSVD_S(const Mat2D &A, Mat2D *S_) {
  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, cv2arma(A), "std");
  AMat2D SS = arma::diagmat(S);
  *S_ = arma2cv(SS);
}

void armaSVD_U(const Mat2D &A, Mat2D *U_) {
  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, cv2arma(A), "std");

  *U_ = arma2cv(U);
}

void armaSVD_V(const Mat2D &A, Mat2D *V_) {
  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, cv2arma(A), "std");

  *V_ = arma2cv(V);
}
