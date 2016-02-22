//
// Created by alexey on 22.02.16.
//

#include <iostream>
#include "perspective_embedding.h"
#include "utils/balls_and_bins.h"

typedef cv::Mat1i Cell;
typedef std::vector<Cell> CellArray;
typedef std::vector<CellArray> CellArray2D;

inline bool all_1i(const cv::Mat& mat) {
  for(int j = 0; j < mat.rows; j++) {
    for(int i = 0; i < mat.cols; i++) {
      if(!mat.at<int>(j, i)) return 0;
    }
  }

  return 1;
}

std::tuple<Mat2DArray, Mat3DArray, Mat4DArray> perspective_embedding(const cv::Mat &data,
                                                                     const unsigned int order,
                                                                     const bool all,
                                                                     const int nargout) {
  const int debug = 0;

  unsigned int K = uint(data.rows),
               N = uint(data.cols);

  auto indices = balls_and_bins(2*order, K, true);

  // For quadratic cases, we have to enforce the two diagonal minor matrices
  // are zero. So we drop all dimensions in the veronese maps whose exponents
  // are larger than order/2.
  auto lastIndices = indices.back();
  if(debug) {
    std::cout << "lastIndices =\n" << lastIndices << std::endl;
  }

  int numDimensions = lastIndices.size().height;
  cv::Mat1i keepDimensions(numDimensions, 1);
  for(int k = 0; k < numDimensions; k++) {
    keepDimensions(k) =
        ((lastIndices(k, 0) + lastIndices(k, 1)) <= order) &&
        ((lastIndices(k, 2) + lastIndices(k, 3)) <= order);
  }

  if(debug) {
    std::cout << "keep =\n" << keepDimensions << std::endl;
  }

  Mat2DArray V(2*order);
  Mat3DArray D(2*order);
  Mat4DArray H(2*order); //if nargout >=3,

  // One column indicates whether there are 0 elements in this data vector.
  cv::Mat1i zeroData = cv::Mat1i::zeros(data.cols, 1);
  cv::Mat1i nonzeroData = cv::Mat1i::zeros(data.cols, 1);
  int zeroFlag = 0;


  for(int i = 0; i < data.cols; i++) {
    for(int j = 0; j < data.rows; j++) {
      zeroData(i) += data.at<double>(j, i) == 0;
    }
    zeroFlag += zeroData(i);

    if(!zeroData(i)) nonzeroData(i) = 1;
  }

  if(debug) {
    std::cout << "zeroData =\n" << zeroData << std::endl;
    std::cout << "nonzeroData =\n" << nonzeroData << std::endl;
    std::cout << "zeroFlag =\n" << zeroFlag << std::endl;
  }

  cv::Mat logData = cv::Mat::zeros(K, N, CV_64F);
  cv::log(data, logData);

  std::vector<cv::Mat> indicesFlt;
  for(int i = 0; i < indices.size(); i++) {
    cv::Mat temp;
    indices[i].convertTo(temp, CV_64F);
    indicesFlt.push_back(temp);
  }

  for(int o = 0; o < 2*order; o++) {
    // Trick to compute the Veronese map using matrix multiplication
    if(all_1i(nonzeroData)) {
      // No exact 0 element in the data, log(Data) is finite, with possible complex terms when the data value is negative
      cv::exp(indicesFlt[o] * logData, V[o]); // (converting) conversion to real dropped

      if(debug) {
        std::cout << "all nonzero" << std::endl;
        std::cout << V[o].size() << std::endl;
      }
    }

    else {
      int rows = indicesFlt[o].rows;
      V[o] = cv::Mat(indicesFlt[o].rows, N, CV_64F);

      for(int dataPoint = 0; dataPoint < N; dataPoint++) {
        if(zeroData(dataPoint) > 0) {
          // data(dataPoint) has 0 elements that are left unprocessed above.
          for(int rowCount=0; rowCount < rows; rowCount++) {
            double prod = 1;
            for(int i = 0; i < data.rows; i++) {
              prod *= std::pow(data.at<double>(i, dataPoint), indicesFlt[o].at<double>(rowCount, i));
            }

            V[o].at<double>(rowCount, dataPoint) = prod;
          }

          if(debug) {
            std::cout << "this zero" << std::endl;
            std::cout << "indicesFlt =\n" << indicesFlt[o] << std::endl;
            std::cout << "logData =\n" << logData.col(dataPoint) << std::endl;
            std::cout << "V =\n" << V[o].col(dataPoint) << std::endl;
          }
        }

        else {
          // (converting) conversion to real dropped
          cv::exp(indicesFlt[o] * logData.col(dataPoint), V[o].col(dataPoint));

          if(debug) {
            std::cout << "this nonzero" << std::endl;
            std::cout << "indicesFlt =\n" << indicesFlt[o] << std::endl;
            std::cout << "logData =\n" << logData.col(dataPoint) << std::endl;
            std::cout << "V =\n" << V[o].col(dataPoint) << std::endl;
          }
        }
      }
    }

    if(o == 0) {
      for(int d = 0; d < K; d++) {
        cv::Mat temp = cv::Mat::zeros(K, N, CV_64F);
        D[o].push_back(temp);
      }

      for(int d = 0; d < K; d++) {
        D[o][d].row(d) = cv::Mat::ones(1, N, CV_64F);
      }

      if(nargout >= 3) {
        for(int t = 0; t < K; t++) {
          for(int d = 0; d < K; d++) {
            cv::Mat temp = cv::Mat::zeros(K, N, CV_64F);
            D[o][t].push_back(temp);
          }
        }
      }
    }

    else {
      //todo
    }
  }

  if(all) {
    return std::make_tuple(V, D, H);
  }

  else {
    //todo
    return std::make_tuple(V, D, H);
  }
}

