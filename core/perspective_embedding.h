// Copyright 2016 Alexey Rozhkov

#ifndef CORE_PERSPECTIVE_EMBEDDING_H_
#define CORE_PERSPECTIVE_EMBEDDING_H_

#include <core/utils/mat_nd.h>
#include <ostream>
#include <vector>


class EmbeddingData {
 private:
  Mat2D veronese;
  Mat3D jacobian;
  Mat4D hessian;

 public:
  EmbeddingData() {}
  EmbeddingData(const EmbeddingData &other);
  EmbeddingData(const EmbeddingData &other, const std::vector<int> &filter);
  EmbeddingData(const EmbeddingInitializer &init, const int N);
  EmbeddingData(const Mat2D &veronese_,
                const Mat3D &jacobian_,
                const Mat4D &hessian_);

  EmbeddingData& operator=(const EmbeddingData& other);
  EmbeddingData& operator=(EmbeddingData&& other);

  const Mat2D &getV() const;
  const Mat3D &getD() const;
  const Mat4D &getH() const;
};

class Embedding {
  // input data dimensions
  const int K, N;

  // embedding order
  const size_t order;

  // copy of data and it's element-wise logarithm
  const Mat2D data, logData;

  // indicates whether there are non-positive values in each column of data
  const IndexMat2D nonPositiveCols;

  // indicates whether there are any non-positive entries in data
  const bool hasNonPositiveVals;

  // the exponents of the veronese map
  const IndexMat2DArray indices;

  // indices in floating point to make matrix types match during multiplication
  const Mat2DArray indicesFlt;

  // arrays of data to hold intermediate results
  const Mat2DArray veroneseArr;
  const Mat3DArray jacobianArr;
  const Mat4DArray hessianArr;

  // veronese mapping
  inline void computeVeroneseMappingMat(Mat2D *out,
                                        const size_t o) const;

  inline void computeVeroneseMappingCol(Mat2D *out,
                                        const size_t o,
                                        const int n) const;

  inline void computeVeroneseMappingElem(Mat2D *out,
                                         const size_t o,
                                         const int n,
                                         const int k) const;

  // derivative calculation
  template<typename T>
  void calcDeriv(const size_t o,
                 const int idx1,
                 const int idx2,
                 const T *Mat1,
                 std::vector<T> *Mat2) const;
  void swapMiddleNondiag(Mat4D *H, const int d, const int h) const;

  void computeVeroneseMapping(const size_t o, Mat2D *vOut) const;
  void computeDerivative(const size_t o, Mat3D *dOut) const;
  void computeHessian(const size_t o, Mat4D *hOut) const;

  Mat2DArray computeVeroneseArr() const;
  Mat3DArray computeJacobianArr() const;
  Mat4DArray computeHessianArr() const;

  EmbeddingData computeEmbedding() const;

  // dimension filtering
  IndexMat2D chooseDimensionsToKeep(const IndexMat2D &lastIndices,
                                    int *numKeepDimensionsOut) const;

 public:
  Embedding(const Mat2D &data, const size_t order);
  EmbeddingData getData() const;
};

std::ostream &operator<<(std::ostream &os, EmbeddingData const &e);

const int Kconst = 5;

EmbeddingData perspective_embedding(const Mat2D &data,
                                    const size_t order);

#endif   // CORE_PERSPECTIVE_EMBEDDING_H_
