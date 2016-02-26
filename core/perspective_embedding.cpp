//
// Created by alexey on 22.02.16.
//

#include <iostream>
#include "perspective_embedding.h"
#include "utils/balls_and_bins.h"


inline bool all_1i(const cv::Mat& mat) {
  for(int j = 0; j < mat.rows; j++) {
    for(int i = 0; i < mat.cols; i++) {
      if(!mat.at<int>(j, i)) return 0;
    }
  }

  return 1;
}

Embedding perspective_embedding(const Mat2D &data,
                                const unsigned int order,
                                const bool all,
                                const int nargout) {
  unsigned int K = uint(data.rows),
               N = uint(data.cols);

  assert(K == Kconst);

  auto indices = balls_and_bins(2*order, K, true);

  // For quadratic cases, we have to enforce the two diagonal minor matrices
  // are zero. So we drop all dimensions in the veronese maps whose exponents
  // are larger than order/2.
  auto lastIndices = indices.back();
  int numDimensions = lastIndices.rows;
  cv::Mat1i keepDimensions(numDimensions, 1);

  int numKeepDimensions = 0;
  for(int k = 0; k < numDimensions; k++) {
    keepDimensions(k) =
        ((lastIndices(k, 0) + lastIndices(k, 1)) <= order) &&
        ((lastIndices(k, 2) + lastIndices(k, 3)) <= order);

    numKeepDimensions += keepDimensions(k);
  }

  Mat2DArray V(2*order);
  Mat3DArray D(2*order);
  Mat4DArray H(2*order);

  for(int o = 0; o < 2*order; o++) {
    int dims = indices[o].rows;
    // (converting) init V
    V[o] = Mat2D::zeros(dims, N);

    // (converting) init D
    D[o] = Mat2DArray(uint(dims));

    for(int d = 0; d < dims; d++) {
      D[o][d] = Mat2D::zeros(K, N);
    }

    // (converting) init H
    if(nargout >= 3) {
      H[o] = Mat3DArray(uint(dims));

      for(int d = 0; d < dims; d++) {
        H[o][d] = Mat2DArray(K);

        for(int k = 0; k < K; k++) {
          H[o][d][k] = Mat2D::zeros(K, N);
        }
      }
    }
  }

  // One column indicates whether there are 0 elements in this data vector.
  cv::Mat1i zeroData = cv::Mat1i::zeros(data.cols, 1);
  cv::Mat1i nonzeroData = cv::Mat1i::zeros(data.cols, 1);

  for(int i = 0; i < data.cols; i++) {
    for(int j = 0; j < data.rows; j++) {
      zeroData(i) += data(j, i) == 0;
    }

    if(!zeroData(i)) nonzeroData(i) = 1;
  }

  Mat2D logData = Mat2D::zeros(K, N);
  cv::log(data, logData);

  Mat2DArray indicesFlt(indices.size());
  for(int i = 0; i < indices.size(); i++) {
    indices[i].convertTo(indicesFlt[i], CV_64F);
  }

  for(int o = 0; o < 2*order; o++) {
    // Trick to compute the Veronese map using matrix multiplication
    if(all_1i(nonzeroData)) {
      // No exact 0 element in the data, log(Data) is finite, with possible complex terms when the data value is negative
      Mat2D temp = indicesFlt[o] * logData;
      cv::exp(temp, V[o]); // (converting) conversion to real dropped
    }

    else {
      int rows = indicesFlt[o].rows;

      for(int dataPoint = 0; dataPoint < N; dataPoint++) {
        if(zeroData(dataPoint) > 0) {
          // data(dataPoint) has 0 elements that are left unprocessed above.
          for(int rowCount = 0; rowCount < rows; rowCount++) {
            V[o](rowCount, dataPoint) = 1;
            for(int i = 0; i < data.rows; i++) {
              V[o](rowCount, dataPoint) *= std::pow(data(i, dataPoint), indicesFlt[o](rowCount, i));
            }
          }
        }

        else {
          // (converting) conversion to real dropped
          Mat2D temp = indicesFlt[o] * logData.col(dataPoint);
          cv::exp(temp, V[o].col(dataPoint));
        }
      }
    }

    if(o == 0) {
      for(int d = 0; d < K; d++) {
        D[o][d].row(d) = Mat2D::ones(1, N);
      }
    }
    else {
      int Mn = indices[o].rows;

      for(int d = 0; d < K; d++) {
        // Take one column of the exponents array of order o
        auto D_indices = indices[o].col(d);
        Mat2D Vd = Mat2D::zeros(Mn, N);

        // Find all of the non-zero exponents, to avoid division by zero
        int nz = 0;
        for(int i = 0; i < D_indices.rows; i++) {
          if(D_indices(i) != 0) {
            V[o-1].row(nz).copyTo(Vd.row(i));
            nz++;
          }
        }

        // Multiply the lower order veronese map by the exponents of the
        // relevant vector element
        for(int j = 0; j < Vd.rows; j++) {
          D[o][j].row(d) = D_indices(j)*Vd.row(j);
        }

        if(nargout >= 3) {
          for(int h = d; h < K; h++) {
            auto H_indices = indices[o].col(h);
            Mat2D Vh = Mat2D::zeros(Mn, N);

            int nz2 = 0;
            for(int j = 0; j < H_indices.rows; j++) {
              if(H_indices(j) != 0) {
                D[o-1][nz2].row(d).copyTo(Vh.row(j));
                nz2++;
              }
            }

            for(int j = 0; j < Vh.rows; j++) {
              H[o][j][d].row(h) = H_indices(j)*Vh.row(j);
            }

            if(d != h) {
              for(int j = 0; j < H[o].size(); j++) {
                for(int i = 0; i < N; i++) {
                  H[o][j][h](d, i) = H[o][j][d](h, i);
                }
              }
            }
          }
        }
      }
    }
  }

  if(all) {
    return std::make_tuple(V, D, H);
  }

  else {
    auto Vlast = V.back();
    auto Dlast = D.back();
    auto Hlast = H.back();

    Mat2D V_keep = Mat2D::zeros(numKeepDimensions, Vlast.cols);
    Mat3D D_keep((uint)numKeepDimensions);  // todo: check difference between uint() and (uint) casts
    Mat4D H_keep((uint)numKeepDimensions);

    int tn = 0;
    for(int j = 0; j < keepDimensions.rows; j++) {
      if(keepDimensions(j)) {
        Vlast.row(j).copyTo(V_keep.row(tn));
        D_keep[tn] = Dlast[j];
        H_keep[tn] = Hlast[j];
        tn++;
      }
    }

    return std::make_tuple(Mat2DArray{V_keep},
                           Mat3DArray{D_keep},
                           Mat4DArray{H_keep});
  }
}