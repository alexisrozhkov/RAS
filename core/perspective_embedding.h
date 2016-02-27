//
// Created by alexey on 22.02.16.
//

#ifndef RAS_PERSPECTIVE_EMBEDDING_H
#define RAS_PERSPECTIVE_EMBEDDING_H

#include <array>
#include <vector>
#include <balls_and_bins.h>
#include <opencv2/core.hpp>


typedef double EmbValT;
typedef cv::Mat_<EmbValT> Mat2D;
typedef std::vector<Mat2D> Mat2DArray;

typedef Mat2DArray Mat3D;
typedef std::vector<Mat3D> Mat3DArray;

typedef Mat3DArray Mat4D;
typedef std::vector<Mat4D> Mat4DArray;

typedef std::array<std::vector<EmbValT>, 3> EmbeddingInitializer;


class Embedding {
  Mat2DArray veronese;
  Mat3DArray jacobian;
  Mat4DArray hessian;

  Mat2D data, logData;

  // indicates whether there are any zero entries in data
  bool hasZeros;

  // indicates whether there are zeros in each column of data
  IndexMat2D zeroCols;

  const int K, N;

  const IndexMat2DArray indices;
  Mat2DArray indicesFlt;

  // veronese mapping
  inline void computeVeroneseMappingMat(Mat2D &out, const int o) const;
  inline void computeVeroneseMappingCol(Mat2D &out, const int o, const int n) const;
  inline void computeVeroneseMappingElem(Mat2D &out, const int o, const int n, const int k) const;
  void computeVeroneseMapping(const int o, Mat2D &vOut) const;

  // derivative calculation
  template<typename T>
  void calcDeriv(const int o, const int idx1, const int idx2, const T &Mat1, std::vector<T> &Mat2) const;
  void swapMiddleNondiag(Mat4D &H, const int d, const int h) const;
  void computeDerivatives(const int o, Mat3D &dOut, Mat4D &hOut) const;

  // dimension filtering
  IndexMat2D chooseDimensionsToKeep(const IndexMat2D &lastIndices, const int order, int &numKeepDimensionsOut) const;
  void filterDimensions(const IndexMat2D &indicesLast, const int order);

 public:
  Embedding(const Mat2D &data, const unsigned int order, const bool filterDims);

  Embedding(const EmbeddingInitializer &init, const int N);

  const Mat2DArray& getV() const;
  const Mat3DArray& getD() const;
  const Mat4DArray& getH() const;
};

const int Kconst = 5;

Embedding perspective_embedding(const Mat2D& data,
                                const unsigned int order,
                                const bool all=false);

#endif //RAS_PERSPECTIVE_EMBEDDING_H
