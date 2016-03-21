// Copyright 2016 Alexey Rozhkov

#include <core/utils/balls_and_bins.h>
#include <core/perspective_embedding.h>
#include <limits>


// utilities
IndexMat2D findNonPositiveCols(const Mat2D &data) {
  IndexMat2D nonPositive = IndexMat2D::zeros(data.cols, 1);

  for (int i = 0; i < data.cols; i++) {
    for (int j = 0; j < data.rows; j++) {
      nonPositive(i) += data(j, i) <= std::numeric_limits<EmbValT>::epsilon();
    }
  }

  return nonPositive;
}

bool checkNonPositiveVals(const Mat2D &nonZeroCols) {
  bool hasNonPositiveVals = false;

  for (int i = 0; i < nonZeroCols.rows; i++) {
    if (nonZeroCols(i) > 0) {
      hasNonPositiveVals = true;
      break;
    }
  }

  return hasNonPositiveVals;
}

inline Mat2D matLog(const Mat2D &mat) {
  Mat2D out;
  cv::log(mat, out);
  return out;
}

inline Mat2DArray mat2dToFloat(const IndexMat2DArray &indices) {
  Mat2DArray out = Mat2DArray(indices.size());
  for (uint i = 0; i < indices.size(); i++) {
    indices[i].convertTo(out[i], CV_64F);
  }
  return out;
}


// Embedding methods
inline void Embedding::computeVeroneseMappingMat(Mat2D *out,
                                                 const int o) const {
  Mat2D temp = indicesFlt[o] * logData;
  cv::exp(temp, *out);
}

inline void Embedding::computeVeroneseMappingCol(Mat2D *out,
                                                 const int o,
                                                 const int n) const {
  Mat2D temp = indicesFlt[o] * logData.col(n);
  cv::exp(temp, out->col(n));
}

inline void Embedding::computeVeroneseMappingElem(Mat2D *out,
                                                  const int o,
                                                  const int n,
                                                  const int k) const {
  (*out)(k, n) = 1;

  for (int i = 0; i < K; i++) {
    (*out)(k, n) *=
        std::pow((EmbValT) data(i, n), (EmbValT) indicesFlt[o](k, i));
  }
}

void Embedding::computeVeroneseMapping(const int o, Mat2D *vOut) const {
  const int dims = indices[o].rows;
  *vOut = Mat2D::zeros(dims, N);

  // Trick to compute the Veronese map using matrix multiplication
  if (!hasNonPositiveVals) {
    // Only positive elements in the data, log(Data) is finite
    computeVeroneseMappingMat(vOut, o);
  } else {
    for (int n = 0; n < N; n++) {
      if (nonPositiveCols(n) > 0) {
        // we have non-positive elements that are left unprocessed above.
        for (int k = 0; k < dims; k++) {
          computeVeroneseMappingElem(vOut, o, n, k);
        }
      } else {
        computeVeroneseMappingCol(vOut, o, n);
      }
    }
  }
}

// hacky way to handle different ways to index Mat2D and Mat3D in a
// similar fashion
template<typename T>
struct SliceMat { };

template<>
struct SliceMat<Mat2D> {
  static Mat2D getSliceRow(const Mat2D &m, const int idx1, const int idx2) {
    return m.row(idx1 == -1 ? idx2 : idx1);
  }
};

template<>
struct SliceMat<Mat3D> {
  static Mat2D getSliceRow(const Mat3D &m, const int idx1, const int idx2) {
    return m[idx1].row(idx2);
  }
};

// calculation of jacobian and hessian values are very similar, except for the
// matrix indexing. this is used here, and for indexing a bit of trait-like
// code is introduced.
// expects Mat2 to be zero-initialized
template<typename T>
void Embedding::calcDeriv(const int o,
                          const int idx1,
                          const int idx2,
                          const T *Mat1,
                          std::vector<T>  // NOLINT(build/include_what_you_use)
                          *Mat2) const {
  const auto D_indices = indices[o].col(idx1);

  // consider only non-zero exponents
  int nz = 0;
  for (int i = 0; i < D_indices.rows; i++) {
    if (D_indices(i) != 0) {
      // Multiply the lower order map by the exponents of the relevant
      // vector element
      SliceMat<T>::getSliceRow((*Mat2)[i], idx2, idx1) =
          D_indices(i) * SliceMat<T>::getSliceRow(*Mat1, nz, idx2);
      nz++;
    }
  }
}

void Embedding::swapMiddleNondiag(Mat4D *H, const int d, const int h) const {
  if (d != h) {
    for (uint j = 0; j < H->size(); j++) {
      for (int i = 0; i < N; i++) {
        (*H)[j][h](d, i) = (*H)[j][d](h, i);
      }
    }
  }
}

void Embedding::computeDerivatives(const int o,
                                   Mat3D *dOut,
                                   Mat4D *hOut) const {
  const int dims = indices[o].rows;

  *dOut = Mat3D_zeros(dims, K, N);
  *hOut = Mat4D_zeros(dims, K, K, N);

  if (o == 0) {
    for (int d = 0; d < K; d++) {
      (*dOut)[d].row(d) = Mat2D::ones(1, N);
    }
  } else {
    for (int d = 0; d < K; d++) {
      calcDeriv(o, d, -1, &(veronese[o - 1]), dOut);

      for (int h = d; h < K; h++) {
        calcDeriv(o, h, d, &(jacobian[o - 1]), hOut);
        swapMiddleNondiag(hOut, d, h);
      }
    }
  }
}

void Embedding::filterDimensions(const IndexMat2D &indicesLast,
                                 const int order) {
  int numKeepDimensions = 0;
  const auto keepDimensions =
      chooseDimensionsToKeep(indicesLast, order, &numKeepDimensions);

  const auto &V = veronese.back();
  const auto &D = jacobian.back();
  const auto &H = hessian.back();

  Mat2D V_keep = Mat2D::zeros(numKeepDimensions, N);
  Mat3D D_keep((uint) numKeepDimensions);
  Mat4D H_keep((uint) numKeepDimensions);

  int kd = 0;
  for (int j = 0; j < keepDimensions.rows; j++) {
    if (keepDimensions(j)) {
      V.row(j).copyTo(V_keep.row(kd));
      D_keep[kd] = D[j];
      H_keep[kd] = H[j];
      kd++;
    }
  }

  // update
  veronese = Mat2DArray{V_keep};
  jacobian = Mat3DArray{D_keep};
  hessian = Mat4DArray{H_keep};
}

// For quadratic cases, we have to enforce that two diagonal minor matrices
// are zero. So we drop all dimensions in the veronese maps whose exponents
// are larger than order/2.
IndexMat2D Embedding::chooseDimensionsToKeep(const IndexMat2D &lastIndices,
                                             const int order,
                                             int *numKeepDimensionsOut) const {
  const int numDimensions = lastIndices.rows;
  IndexMat2D keepDimensions(numDimensions, 1);

  *numKeepDimensionsOut = 0;
  for (int k = 0; k < numDimensions; k++) {
    keepDimensions(k) =
        ((lastIndices(k, 0) + lastIndices(k, 1)) <= order) &&
            ((lastIndices(k, 2) + lastIndices(k, 3)) <= order);

    *numKeepDimensionsOut += keepDimensions(k);
  }

  return keepDimensions;
}

Embedding::Embedding(const Mat2D &data_,
                     const unsigned int order,
                     const bool filterDims) :
    K(data_.rows),
    N(data_.cols),

    veronese(2 * order),
    jacobian(2 * order),
    hessian(2 * order),

    data(data_),
    logData(matLog(data_)),
    nonPositiveCols(findNonPositiveCols(data_)),
    hasNonPositiveVals(checkNonPositiveVals(nonPositiveCols)),

    indices(balls_and_bins(2 * order, (uint) K, true)),
    indicesFlt(mat2dToFloat(indices)) {
  for (uint o = 0; o < 2 * order; o++) {
    computeVeroneseMapping(o, &(veronese[o]));
    computeDerivatives(o, &(jacobian[o]), &(hessian[o]));
  }

  if (filterDims) filterDimensions(indices.back(), order);
}

Embedding::Embedding(const EmbeddingInitializer &init, const int N_) :
    K(Kconst),
    N(N_),
    hasNonPositiveVals(false) {  // true of false doesn't matter here
  const int dims = static_cast<int>(init[0].size()) / N;

  Mat2D V = Mat2D(init[0]).reshape(0, dims);

  Mat3D D((uint) dims);
  for (int k = 0; k < dims; k++) {
    D[k] = Mat2D(K, N);

    for (int j = 0; j < K; j++) {
      for (int i = 0; i < N; i++) {
        D[k](j, i) = init[1][(i * dims + k) * K + j];
      }
    }
  }

  Mat4D H((uint) dims);

  for (int l = 0; l < dims; l++) {
    H[l] = Mat3D((uint) K);

    for (int k = 0; k < K; k++) {
      H[l][k] = Mat2D(K, N);

      for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) {
          H[l][k](j, i) = init[2][((i * K + j) * dims + l) * K + k];
        }
      }
    }
  }

  veronese.push_back(V);
  jacobian.push_back(D);
  hessian.push_back(H);
}

const Mat2D &Embedding::getV() const {
  return veronese.back();
}

const Mat3D &Embedding::getD() const {
  return jacobian.back();
}

const Mat4D &Embedding::getH() const {
  return hessian.back();
}


Embedding perspective_embedding(const Mat2D &data,
                                const unsigned int order,
                                const bool all) {
  assert(data.rows == Kconst);

  return Embedding(data, order, !all);
}

std::ostream &operator<<(std::ostream &os, Embedding const &e) {
  os << e.getV() << std::endl << std::endl;
  os << e.getD() << std::endl;
  os << e.getH() << std::endl;

  return os;
}
