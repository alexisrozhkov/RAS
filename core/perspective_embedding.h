// Copyright 2016 Alexey Rozhkov

#ifndef CORE_PERSPECTIVE_EMBEDDING_H_
#define CORE_PERSPECTIVE_EMBEDDING_H_

#include <core/utils/mat_nd.h>
#include <ostream>
#include <vector>

class EmbeddingData {
 protected:
  Mat2DArray veronese;
  Mat3DArray jacobian;
  Mat4DArray hessian;

  EmbeddingData(const unsigned int order);

 public:
  EmbeddingData(const Mat2D &veronese,
                const Mat3D &jacobian,
                const Mat4D &hessian);

  EmbeddingData(const EmbeddingInitializer &init,
                const int N);

  const Mat2D &getV() const;
  const Mat3D &getD() const;
  const Mat4D &getH() const;
};

class Embedding : public EmbeddingData {
  // input data dimensions
  const int K, N;

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

  // veronese mapping
  inline void computeVeroneseMappingMat(Mat2D *out,
                                        const int o) const;

  inline void computeVeroneseMappingCol(Mat2D *out,
                                        const int o,
                                        const int n) const;

  inline void computeVeroneseMappingElem(Mat2D *out,
                                         const int o,
                                         const int n,
                                         const int k) const;

  void computeVeroneseMapping(const int o, Mat2D *vOut) const;

  // derivative calculation
  template<typename T>
  void calcDeriv(const int o,
                 const int idx1,
                 const int idx2,
                 const T *Mat1,
                 std::vector<T> *Mat2) const;
  void swapMiddleNondiag(Mat4D *H, const int d, const int h) const;
  void computeDerivatives(const int o, Mat3D *dOut, Mat4D *hOut) const;

  // dimension filtering
  IndexMat2D chooseDimensionsToKeep(const IndexMat2D &lastIndices,
                                    const int order,
                                    int *numKeepDimensionsOut) const;
  void filterDimensions(const IndexMat2D &indicesLast, const int order);

 public:
  Embedding(const Mat2D &data,
            const unsigned int order,
            const bool filterDims);
};

std::ostream &operator<<(std::ostream &os, EmbeddingData const &e);

const int Kconst = 5;

Embedding perspective_embedding(const Mat2D &data,
                                const unsigned int order,
                                const bool all = false);

#endif   // CORE_PERSPECTIVE_EMBEDDING_H_
