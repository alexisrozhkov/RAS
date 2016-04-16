// Copyright 2016 Alexey Rozhkov

#include <core/utils/arma_wrapper.h>

#define ARMA_DONT_USE_WRAPPER
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

Mat2D arma2cv(const AMat2D &src) {
  // transpose is needed here not only to switch back to row-major, but also
  // to copy the data, because when cv::Mat is constructed from provided data
  // it doesn't copy it, and it seems that there is no flag to force copying
  return Mat2D(static_cast<int>(src.n_cols),
               static_cast<int>(src.n_rows),
               const_cast<EmbValT*>(src.memptr())).t();
}

Mat2D SVD_S(const Mat2D &A) {
  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, cv2arma(A), "std");
  return arma2cv(arma::diagmat(S));
}

Mat2D SVD_U(const Mat2D &A) {
  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, cv2arma(A), "std");
  return arma2cv(U);
}

Mat2D SVD_V(const Mat2D &A) {
  AMat1D S;
  AMat2D U, V;
  arma::svd(U, S, V, cv2arma(A), "std");
  return arma2cv(V);
}

void generalizedSchur(const Mat2D &A,
                      const Mat2D &B,
                      Mat2D *al,
                      Mat2D *be) {
    AMat2D AA, BB, Q, Z;

    arma::qz(AA, BB, Q, Z, cv2arma(A), cv2arma(B));

    *al = arma2cv(arma::diagvec(AA));
    *be = arma2cv(arma::diagvec(BB));
}
