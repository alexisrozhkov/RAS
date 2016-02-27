//
// Created by alexey on 22.02.16.
//

#ifndef RAS_PERSPECTIVE_EMBEDDING_H
#define RAS_PERSPECTIVE_EMBEDDING_H

#include <vector>
#include <opencv2/core.hpp>

typedef cv::Mat1d Mat2D;
typedef std::vector<Mat2D> Mat2DArray;

typedef Mat2DArray Mat3D;
typedef std::vector<Mat3D> Mat3DArray;

typedef Mat3DArray Mat4D;
typedef std::vector<Mat4D> Mat4DArray;

class Embedding {
  Mat2DArray veronese;
  Mat3DArray jacobian;
  Mat4DArray hessian;

  Mat2DArray indicesFlt;

  Mat2D logData;

  // indicates whether there are any zero entries in data
  bool hasZeros;

  // indicates whether there are zeros in each column of data
  cv::Mat1i zeroData;

  void computeDerivatives(const cv::Mat1i &indices,
                          const int o,
                          const int K,
                          const int N);

  void filterDimensions(const cv::Mat1i &indicesLast,
                        const int order);

 public:
  Embedding(const int N,
            const int K,
            const unsigned int order,
            const Mat2D &data,
            const std::vector<cv::Mat1i> &indices,
            const bool filterDims);

  Embedding(const Mat2DArray &V,
            const Mat3DArray &D,
            const Mat4DArray &H);

  const Mat2DArray& getV() const;
  const Mat3DArray& getD() const;
  const Mat4DArray& getH() const;
};

const int Kconst = 5;

Embedding perspective_embedding(const Mat2D& data,
                                const unsigned int order,
                                const bool all=false);

#endif //RAS_PERSPECTIVE_EMBEDDING_H
