//
// Created by alexey on 22.02.16.
//

#include <balls_and_bins.h>
#include "perspective_embedding.h"


Mat3D Mat3D_zeros(const int A,
                  const int B,
                  const int C) {
  auto temp = Mat2DArray(uint(A));

  for(int k = 0; k < A; k++) {
    temp[k] = Mat2D::zeros(B, C);
  }

  return temp;
}

Mat4D Mat4D_zeros(const int A,
                  const int B,
                  const int C,
                  const int D) {
  auto temp = Mat3DArray(uint(A));

  for(int k = 0; k < A; k++) {
    temp[k] = Mat3D_zeros(B, C, D);
  }

  return temp;
}

Embedding zeroEmbedding(const int N,
                        const int K,
                        const unsigned int order,
                        const std::vector<cv::Mat1i> &indices) {
  Mat2DArray V(2*order);
  Mat3DArray D(2*order);
  Mat4DArray H(2*order);

  for(int o = 0; o < 2*order; o++) {
    const int dims = indices[o].rows;

    V[o] = Mat2D::zeros(dims, N);
    D[o] = Mat3D_zeros(dims, K, N);
    H[o] = Mat4D_zeros(dims, K, K, N);
  }

  return std::make_tuple(V, D, H);
}

// For quadratic cases, we have to enforce that two diagonal minor matrices
// are zero. So we drop all dimensions in the veronese maps whose exponents
// are larger than order/2.
cv::Mat1i chooseDimensionsToKeep(const cv::Mat1i &lastIndices,
                                 const int order,
                                 int &numKeepDimensionsOut) {
  const int numDimensions = lastIndices.rows;
  cv::Mat1i keepDimensions(numDimensions, 1);

  numKeepDimensionsOut = 0;
  for(int k = 0; k < numDimensions; k++) {
    keepDimensions(k) =
        ((lastIndices(k, 0) + lastIndices(k, 1)) <= order) &&
        ((lastIndices(k, 2) + lastIndices(k, 3)) <= order);

    numKeepDimensionsOut += keepDimensions(k);
  }

  return keepDimensions;
}

Embedding filterDimensions(const Embedding &e,
                           const cv::Mat1i &indicesLast,
                           const int order) {
  const auto &V = std::get<0>(e).back();
  const auto &D = std::get<1>(e).back();
  const auto &H = std::get<2>(e).back();

  int numKeepDimensions = 0;
  const auto keepDimensions = chooseDimensionsToKeep(indicesLast, order, numKeepDimensions);

  Mat2D V_keep = Mat2D::zeros(numKeepDimensions, V.cols);
  Mat3D D_keep((uint)numKeepDimensions);  // todo: check difference between uint() and (uint) casts
  Mat4D H_keep((uint)numKeepDimensions);

  int tn = 0;
  for(int j = 0; j < keepDimensions.rows; j++) {
    if(keepDimensions(j)) {
      V.row(j).copyTo(V_keep.row(tn));
      D_keep[tn] = D[j];
      H_keep[tn] = H[j];
      tn++;
    }
  }

  return std::make_tuple(Mat2DArray{V_keep},
                         Mat3DArray{D_keep},
                         Mat4DArray{H_keep});
}

cv::Mat1i zeroCols(const Mat2D &data,
                   bool &hasZeros) {
  hasZeros = false;
  cv::Mat1i zeroData = cv::Mat1i::zeros(data.cols, 1);

  for(int i = 0; i < data.cols; i++) {
    for(int j = 0; j < data.rows; j++) {
      zeroData(i) += data(j, i) == 0;  // todo: maybe replace exact comparison with some epsilon
    }

    if(zeroData(i) > 0) hasZeros = true;
  }

  return zeroData;
}

template<typename T> struct SliceMat {};

template<>
struct SliceMat<Mat2D> {
  static cv::Mat_<double> getSliceRow(const Mat2D &Mat1, const int idx1, const int idx2)   {
    return Mat1.row(idx1 == -1 ? idx2 : idx1);
  }
};

template<>
struct SliceMat<Mat3D> {
  static cv::Mat_<double> getSliceRow(const Mat3D &Mat1, const int idx1, const int idx2)   {
    return Mat1[idx1].row(idx2);
  }
};

template<typename T>
void calcDeriv(const cv::Mat1i &indices,
               const int idx1,
               const int idx2,
               const int N,
               const T &Mat1,
               std::vector<T> &Mat2) {
  auto D_indices = indices.col(idx1);
  Mat2D Vd = Mat2D::zeros(indices.rows, N);

  // Find all of the non-zero exponents, to avoid division by zero
  int nz = 0;
  for(int i = 0; i < D_indices.rows; i++) {
    if(D_indices(i) != 0) {
      SliceMat<T>::getSliceRow(Mat1, nz, idx2).copyTo(Vd.row(i));
      nz++;
    }
  }

  // Multiply the lower order veronese map by the exponents of the
  // relevant vector element
  for(int j = 0; j < Vd.rows; j++) {
    SliceMat<T>::getSliceRow(Mat2[j], idx2, idx1) = D_indices(j)*Vd.row(j);
  }
}

void swapMiddleNondiag(Mat4D &H,
                       const int N,
                       const int d,
                       const int h) {
  if(d != h) {
    for(int j = 0; j < H.size(); j++) {
      for(int i = 0; i < N; i++) {
        H[j][h](d, i) = H[j][d](h, i);
      }
    }
  }
}

void computeDerivatives(Embedding &e,
                        const cv::Mat1i &indices,
                        const int o,
                        const int K,
                        const int N) {
  const auto &V = std::get<0>(e);
  auto &D = std::get<1>(e);
  auto &H = std::get<2>(e);

  if(o == 0) {
    for(int d = 0; d < K; d++) {
      D[o][d].row(d) = Mat2D::ones(1, N);
    }
  }
  else {
    for(int d = 0; d < K; d++) {
      // Take one column of the exponents array of order o
      calcDeriv(indices, d, -1, N, V[o - 1], D[o]);

      //if(nargout >= 3) {
        for(int h = d; h < K; h++) {
          calcDeriv(indices, h, d, N, D[o - 1], H[o]);
          swapMiddleNondiag(H[o], N, d, h);
        }
      //}
    }
  }
}

// calc veronese map for whole matrix
inline void calcVeroneseMat(const Mat2D &indicesFlt,
                            const Mat2D &logData,
                            Mat2D &out) {
  Mat2D temp = indicesFlt * logData;
  cv::exp(temp, out); // (converting) conversion to real dropped
}

// calc veronese map for specified column
inline void calcVeroneseCol(const int dataPoint,
                            const Mat2D &indicesFlt,
                            const Mat2D &logData,
                            Mat2D &out) {
  Mat2D temp = indicesFlt * logData.col(dataPoint);
  cv::exp(temp, out.col(dataPoint)); // (converting) conversion to real dropped
}

// calc veronese map for specified element
inline void calcVeroneseElem(const int rowCount,
                             const int dataPoint,
                             const Mat2D &indicesFlt,
                             const Mat2D &data,
                             Mat2D &out) {
  out(rowCount, dataPoint) = 1;
  for(int i = 0; i < data.rows; i++) {
    out(rowCount, dataPoint) *= std::pow(data(i, dataPoint), indicesFlt(rowCount, i));
  }
}

void computeVeroneseMapping(const Mat2D &indicesFlt,
                            const Mat2D &logData,
                            const Mat2D &data,
                            const Mat2D &zeroData,
                            const bool hasZeros,
                            Mat2D &out) {
  // Trick to compute the Veronese map using matrix multiplication
  if(!hasZeros) {
    // No exact 0 element in the data, log(Data) is finite, with possible complex terms when the data value is negative
    calcVeroneseMat(indicesFlt, logData, out);
  }

  else {
    for(int dataPoint = 0; dataPoint < logData.cols; dataPoint++) {
      if(zeroData(dataPoint) > 0) {
        // data(dataPoint) has 0 elements that are left unprocessed above.
        for(int rowCount = 0; rowCount < indicesFlt.rows; rowCount++) {
          calcVeroneseElem(rowCount, dataPoint, indicesFlt, data, out);
        }
      }

      else {
        calcVeroneseCol(dataPoint, indicesFlt, logData, out);
      }
    }
  }
}

Embedding perspective_embedding(const Mat2D &data,
                                const unsigned int order,
                                const bool all) {
  const unsigned int K = uint(data.rows),
                     N = uint(data.cols);

  assert(K == Kconst);

  const auto indices = balls_and_bins(2*order, K, true);

  Mat2DArray indicesFlt(indices.size());
  for(int i = 0; i < indices.size(); i++) {
    indices[i].convertTo(indicesFlt[i], CV_64F);
  }

  // One column indicates whether there are 0 elements in this data vector.
  bool hasZeros = false;
  const auto zeroData = zeroCols(data, hasZeros);

  // calc element-wise logarithm of data for faster Veronese mapping computation
  Mat2D logData = Mat2D::zeros(K, N);
  cv::log(data, logData);

  Embedding embedding = zeroEmbedding(N, K, order, indices);
  auto &V = std::get<0>(embedding);

  for(int o = 0; o < 2*order; o++) {
    computeVeroneseMapping(indicesFlt[o], logData, data, zeroData, hasZeros, V[o]);
    computeDerivatives(embedding, indices[o], o, K, N);
  }

  if(all) {
    return embedding;
  }

  else {
    return filterDimensions(embedding, indices.back(), order);
  }
}