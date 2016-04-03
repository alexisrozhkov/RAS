// Copyright 2016 Alexey Rozhkov

#include <core/utils/balls_and_bins.h>
#include <core/perspective_embedding.h>

#include <limits>
#include <vector>


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
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i].convertTo(out[i], CV_64F);
  }
  return out;
}


// Embedding methods
inline void Embedding::computeVeroneseMappingMat(Mat2D *out,
                                                 const size_t o) const {
  Mat2D temp = indicesFlt[o] * logData;
  cv::exp(temp, *out);
}

inline void Embedding::computeVeroneseMappingCol(Mat2D *out,
                                                 const size_t o,
                                                 const int n) const {
  Mat2D temp = indicesFlt[o] * logData.col(n);
  cv::exp(temp, out->col(n));
}

inline void Embedding::computeVeroneseMappingElem(Mat2D *out,
                                                  const size_t o,
                                                  const int n,
                                                  const int k) const {
  (*out)(k, n) = 1;

  for (int i = 0; i < K; i++) {
    (*out)(k, n) *=
        std::pow((EmbValT) data(i, n), (EmbValT) indicesFlt[o](k, i));
  }
}

void Embedding::computeVeroneseMapping(const size_t o, Mat2D *vOut) const {
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
void Embedding::calcDeriv(const size_t o,
                          const int idx1,
                          const int idx2,
                          const T *Mat1,
                          std::vector<T> *Mat2) const {
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
    for (size_t j = 0; j < H->size(); j++) {
      for (int i = 0; i < N; i++) {
        (*H)[j][h](d, i) = (*H)[j][d](h, i);
      }
    }
  }
}

void Embedding::computeDerivative(const size_t o,
                                  Mat3D *dOut) const {
  const int dims = indices[o].rows;

  *dOut = Mat3D_zeros(dims, K, N);

  if (o == 0) {
    for (int d = 0; d < K; d++) {
      (*dOut)[d].row(d) = Mat2D::ones(1, N);
    }
  } else {
    for (int d = 0; d < K; d++) {
      calcDeriv(o, d, -1, &(veroneseArr[o - 1]), dOut);
    }
  }
}

void Embedding::computeHessian(const size_t o,
                               Mat4D *hOut) const {
  const int dims = indices[o].rows;
  *hOut = Mat4D_zeros(dims, K, K, N);

  if (o != 0) {
    for (int d = 0; d < K; d++) {
      for (int h = d; h < K; h++) {
        calcDeriv(o, h, d, &(jacobianArr[o - 1]), hOut);
        swapMiddleNondiag(hOut, d, h);
      }
    }
  }
}

// For quadratic cases, we have to enforce that two diagonal minor matrices
// are zero. So we drop all dimensions in the veronese maps whose exponents
// are larger than order/2.
IndexMat2D Embedding::chooseDimensionsToKeep(const IndexMat2D &lastIndices,
                                             int *numKeepDimensionsOut) const {
  const int numDimensions = lastIndices.rows;
  IndexMat2D keepDimensions(numDimensions, 1);

  *numKeepDimensionsOut = 0;
  for (int k = 0; k < numDimensions; k++) {
    keepDimensions(k) =
        ((lastIndices(k, 0) + lastIndices(k, 1)) <= static_cast<int>(order)) &&
        ((lastIndices(k, 2) + lastIndices(k, 3)) <= static_cast<int>(order));

    *numKeepDimensionsOut += keepDimensions(k);
  }

  return keepDimensions;
}

Mat2DArray Embedding::computeVeroneseArr() const {
  Mat2DArray veroneseArr_(2 * order);

  for (size_t o = 0; o < 2 * order; o++) {
    computeVeroneseMapping(o, &(veroneseArr_[o]));
  }

  return veroneseArr_;
}

Mat3DArray Embedding::computeJacobianArr() const {
  Mat3DArray jacobianArr_(2 * order);

  for (size_t o = 0; o < 2 * order; o++) {
    computeDerivative(o, &(jacobianArr_[o]));
  }

  return jacobianArr_;
}

Mat4DArray Embedding::computeHessianArr() const {
  Mat4DArray hessianArr_(2 * order);

  for (size_t o = 0; o < 2 * order; o++) {
    computeHessian(o, &(hessianArr_[o]));
  }

  return hessianArr_;
}

EmbeddingData Embedding::computeEmbedding() const {
  int numKeepDimensions = 0;
  const auto keepDimensions =
      chooseDimensionsToKeep(indices.back(), &numKeepDimensions);

  const auto &V = veroneseArr.back();
  const auto &D = jacobianArr.back();
  const auto &H = hessianArr.back();

  Mat2D veronese = Mat2D::zeros(numKeepDimensions, N);
  Mat3D jacobian = Mat3D((size_t) numKeepDimensions);
  Mat4D hessian = Mat4D((size_t) numKeepDimensions);

  int kd = 0;
  for (int j = 0; j < keepDimensions.rows; j++) {
    if (keepDimensions(j)) {
      V.row(j).copyTo(veronese.row(kd));
      jacobian[kd] = D[j];
      hessian[kd] = H[j];
      kd++;
    }
  }

  return EmbeddingData(veronese, jacobian, hessian);
}

Embedding::Embedding(const Mat2D &data_,
                     const size_t o) :
    K(data_.rows),
    N(data_.cols),
    order(o),

    data(data_),
    logData(matLog(data_)),
    nonPositiveCols(findNonPositiveCols(data_)),
    hasNonPositiveVals(checkNonPositiveVals(nonPositiveCols)),

    indices(balls_and_bins(2 * order, static_cast<size_t>(K), true)),
    indicesFlt(mat2dToFloat(indices)),

    veroneseArr(computeVeroneseArr()),
    jacobianArr(computeJacobianArr()),
    hessianArr(computeHessianArr()) {
}

EmbeddingData Embedding::getData() const {
  return computeEmbedding();
}

EmbeddingData::EmbeddingData(const EmbeddingInitializer &init, const int N) {
  const int K = Kconst;
  const int dims = static_cast<int>(init[0].size()) / N;

  veronese = Mat2D(init[0]).reshape(0, dims);

  jacobian = Mat3D((size_t) dims);

  for (int k = 0; k < dims; k++) {
    jacobian[k] = Mat2D(K, N);

    for (int j = 0; j < K; j++) {
      for (int i = 0; i < N; i++) {
        jacobian[k](j, i) = init[1][(i * dims + k) * K + j];
      }
    }
  }

  hessian = Mat4D((size_t) dims);

  for (int l = 0; l < dims; l++) {
    hessian[l] = Mat3D((size_t) K);

    for (int k = 0; k < K; k++) {
      hessian[l][k] = Mat2D(K, N);

      for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) {
          hessian[l][k](j, i) = init[2][((i * K + j) * dims + l) * K + k];
        }
      }
    }
  }
}

EmbeddingData::EmbeddingData(const Mat2D &veronese_,
                             const Mat3D &jacobian_,
                             const Mat4D &hessian_) :
    veronese(Mat2D_clone(veronese_)),
    jacobian(Mat3D_clone(jacobian_)),
    hessian(Mat4D_clone(hessian_)) {
}

EmbeddingData::EmbeddingData(const EmbeddingData &other) :
    veronese(Mat2D_clone(other.veronese)),
    jacobian(Mat3D_clone(other.jacobian)),
    hessian(Mat4D_clone(other.hessian)) {
}

EmbeddingData::EmbeddingData(const EmbeddingData &other,
                             const std::vector<int> &filter) {
  veronese = filterIdx2(other.getV(), filter);
  jacobian = filterIdx3(other.getD(), filter);
  hessian = filterIdx4(other.getH(), filter);
}

EmbeddingData& EmbeddingData::operator=(const EmbeddingData& other) {
  if (this != &other) {
    veronese = Mat2D_clone(other.veronese);
    jacobian = Mat3D_clone(other.jacobian);
    hessian = Mat4D_clone(other.hessian);
  }

  return *this;
}

EmbeddingData& EmbeddingData::operator=(EmbeddingData&& other) {
  veronese = std::move(other.veronese);
  jacobian = std::move(other.jacobian);
  hessian = std::move(other.hessian);

  return *this;
}

const Mat2D &EmbeddingData::getV() const {
  return veronese;
}

const Mat3D &EmbeddingData::getD() const {
  return jacobian;
}

const Mat4D &EmbeddingData::getH() const {
  return hessian;
}

EmbeddingData perspective_embedding(const Mat2D &data,
                                    const size_t order) {
  assert(data.rows == Kconst);

  return Embedding(data, order).getData();
}

std::ostream &operator<<(std::ostream &os, EmbeddingData const &e) {
  os << e.getV() << std::endl << std::endl;
  os << e.getD() << std::endl;
  os << e.getH() << std::endl;

  return os;
}
