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

std::tuple<Mat2DArray, Mat3DArray, Mat4DArray> perspective_embedding(const Mat2D &data,
                                                                     const unsigned int order,
                                                                     const bool all,
                                                                     const int nargout) {
  unsigned int K = uint(data.rows),
               N = uint(data.cols);

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
  Mat4DArray H(2*order); //if nargout >=3,

  for(int o = 0; o < 2*order; o++) {
    int dims = indices[o].rows;
    // (converting) init V
    V[o] = Mat2D::zeros(dims, N);

    // (converting) init D
    D[o] = Mat2DArray(uint(dims));

    for(int d = 0; d < dims; d++) {
      D[o][d] = Mat2D::zeros(K, N);
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

      if(nargout >= 3) {
        //todo
        /*
        for(int t = 0; t < K; t++) {
          for(int d = 0; d < K; d++) {
            cv::Mat temp = cv::Mat::zeros(K, N, CV_64F);
            H[o][t].push_back(temp);
          }
        }
         */
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
      }
    }
  }

  if(all) {
    return std::make_tuple(V, D, H);
  }

  else {
    auto Vlast = V.back();
    auto Dlast = D.back();
    //auto Hlast = H.back();

    Mat2D V_keep = Mat2D::zeros(numKeepDimensions, Vlast.cols);
    Mat3D D_keep((unsigned int)numKeepDimensions);//todo: check difference between uint and unsigned int

    int tn = 0;
    for(int j = 0; j < keepDimensions.rows; j++) {
      if(keepDimensions(j)) {
        Vlast.row(j).copyTo(V_keep.row(tn));
        D_keep[tn] = Dlast[j];
        tn++;
      }
    }
    Mat2DArray Vvec;
    Mat3DArray Dvec;
    Mat4DArray Hvec;

    Vvec.push_back(V_keep);
    Dvec.push_back(D_keep);

    //todo
    //Hvec.push_back(H_keep);

    return std::make_tuple(Vvec, Dvec, Hvec);
  }
}