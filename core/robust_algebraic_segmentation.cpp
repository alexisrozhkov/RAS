// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <core/perspective_embedding.h>
#include <core/utils/math.h>
#include <core/utils/arma_wrapper.h>
#include <core/utils/subspace_angle.h>

#include <vector>
#include <limits>
#include <utility>
#include <iostream>
#include <algorithm>
#include <tuple>

static const int HSampleSize = 4;
static const int FSampleSize = 8;
static const EmbValT FThreshold = 0.035;  // FThreshold = 0.015;
static const EmbValT HAngleThreshold = 0.04;

static const int dimensionCount = 5;

// utils

int n1_8choose4[] = {
  1, 2, 3, 4,
  1, 2, 3, 5,
  1, 2, 3, 6,
  1, 2, 3, 7,
  1, 2, 3, 8,
  1, 2, 4, 5,
  1, 2, 4, 6,
  1, 2, 4, 7,
  1, 2, 4, 8,
  1, 2, 5, 6,
  1, 2, 5, 7,
  1, 2, 5, 8,
  1, 2, 6, 7,
  1, 2, 6, 8,
  1, 2, 7, 8,
  1, 3, 4, 5,
  1, 3, 4, 6,
  1, 3, 4, 7,
  1, 3, 4, 8,
  1, 3, 5, 6,
  1, 3, 5, 7,
  1, 3, 5, 8,
  1, 3, 6, 7,
  1, 3, 6, 8,
  1, 3, 7, 8,
  1, 4, 5, 6,
  1, 4, 5, 7,
  1, 4, 5, 8,
  1, 4, 6, 7,
  1, 4, 6, 8,
  1, 4, 7, 8,
  1, 5, 6, 7,
  1, 5, 6, 8,
  1, 5, 7, 8,
  1, 6, 7, 8,
  2, 3, 4, 5,
  2, 3, 4, 6,
  2, 3, 4, 7,
  2, 3, 4, 8,
  2, 3, 5, 6,
  2, 3, 5, 7,
  2, 3, 5, 8,
  2, 3, 6, 7,
  2, 3, 6, 8,
  2, 3, 7, 8,
  2, 4, 5, 6,
  2, 4, 5, 7,
  2, 4, 5, 8,
  2, 4, 6, 7,
  2, 4, 6, 8,
  2, 4, 7, 8,
  2, 5, 6, 7,
  2, 5, 6, 8,
  2, 5, 7, 8,
  2, 6, 7, 8,
  3, 4, 5, 6,
  3, 4, 5, 7,
  3, 4, 5, 8,
  3, 4, 6, 7,
  3, 4, 6, 8,
  3, 4, 7, 8,
  3, 5, 6, 7,
  3, 5, 6, 8,
  3, 5, 7, 8,
  3, 6, 7, 8,
  4, 5, 6, 7,
  4, 5, 6, 8,
  4, 5, 7, 8,
  4, 6, 7, 8,
  5, 6, 7, 8
};

const IndexMat2D binomMat = IndexMat2D(70, 4, n1_8choose4);



typedef std::pair<EmbValT, int> IndexedVal;
typedef std::vector<IndexedVal> IndexedVec;

inline void sortIndexedVec(IndexedVec *influence) {
  sort(influence->begin(),
       influence->end(),
       [](const IndexedVal& a, const IndexedVal& b)
       { return (a.first) < (b.first); });
}

template <class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator copy_idx_if(InputIterator first, InputIterator last,
                        OutputIterator result, UnaryPredicate pred) {
  InputIterator start = first;

  while (first != last) {
    if (pred(*first)) {
      *result = first-start;
      ++result;
    }
    ++first;
  }
  return result;
}

template <class InputType, class UnaryPredicate>
std::vector<int> filterIndices(const InputType &in, UnaryPredicate pred) {
  std::vector<int> out(in.end() - in.begin());

  auto it = copy_idx_if(in.begin(), in.end(), out.begin(), pred);

  out.resize(it - out.begin());

  return out;
}

// ras stuff

Mat3D polyHessian(const Mat2D &coeffs,
                  const Mat4D &hessian) {
  const int samplesCount = hessian[0][0].cols;
  Mat3D hpn = Mat3D_zeros(dimensionCount, dimensionCount, samplesCount);

  // iterate over right upper triangular part
  for (int i = 0; i < dimensionCount; i++) {
    for (int j = i; j < dimensionCount; j++) {
      hpn[i].row(j) = coeffs.t() * sliceIdx23(hessian, i, j);

      // copy elements to the left bottom triangular part
      if (i != j) {
        hpn[i].row(j).copyTo(hpn[j].row(i));
      }
    }
  }

  return hpn;
}

Mat2D clusterQuadratic(const int clusterIdx,
                       const IndexMat2D &labels,
                       const EmbeddingData &emb,
                       const bool NORMALIZE_QUADRATICS) {
  const std::vector<int> currIndices =
      filterIndices(labels, [clusterIdx](int val){return val == clusterIdx;});

  const auto clusterData = filterIdx2(emb.getV(), currIndices);
  const auto clusterHessian = filterIdx4(emb.getH(), currIndices);

  const Mat2D U = SVD_U(clusterData);

  const Mat3D hpn = polyHessian(U.col(U.cols - 1),
                                clusterHessian);

  Mat2D quad = meanIdx3(hpn);
  if (NORMALIZE_QUADRATICS) {
    quad /= cv::norm(quad);
  }

  return quad;
}

inline EmbValT ptQuadDist(const Mat2D &u1, const Mat2D &quad) {
  Mat2D val = u1.t() * quad * u1;
  return fabs(val(0));
}

void mergeClusters(const unsigned int groupCount,
                   const bool NORMALIZE_QUADRATICS,
                   const Mat2D &jointImageData,
                   const EmbeddingData &emb,
                   IndexMat2D *labelsP,
                   Mat3D *quadraticsP) {
  IndexMat2D &labels = *labelsP;
  Mat3D &quadratics = *quadraticsP;

  int clusterCount = static_cast<int>(quadratics.size());

  while (clusterCount > static_cast<int>(groupCount)) {
    // count number of points in clusters
    std::vector<int> num(static_cast<size_t>(clusterCount));
    for (int i = 0; i < labels.cols; i++) {
      num[labels(i)] += 1;
    }

    // find smallest cluster
    int smallestClust = std::min_element(num.begin(), num.end()) - num.begin();

    // swap last cluster with smallest
    if (smallestClust != clusterCount-1) {
      for (int i = 0; i < labels.cols; i++) {
        // swap labels
        if (labels(i) == smallestClust) {
          labels(i) = clusterCount-1;
        } else if (labels(i) == clusterCount-1) {
          labels(i) = smallestClust;
        }
      }

      // swap quadratics
      std::swap(quadratics[smallestClust], quadratics[clusterCount-1]);
    }

    std::vector<int> clusterChanged;

    // reassign points from last(smallest) cluster to other clusters
    for (int j = 0; j < labels.cols; j++) {
      if (labels(j) != clusterCount - 1) {
        continue;
      }

      Mat2D pt = jointImageData.col(j);

      // find closest cluster
      int minIdx = 0;
      EmbValT minDist = std::numeric_limits<EmbValT>::max();

      for (int clstrIdx = 0; clstrIdx < clusterCount - 1; clstrIdx++) {
        EmbValT distance = ptQuadDist(pt, quadratics[clstrIdx]);

        if (distance < minDist) {
          minDist = distance;
          minIdx = clstrIdx;
        }
      }

      // move points from last cluster to the closest, mark it as updated
      labels(j) = minIdx;
      clusterChanged.push_back(minIdx);
    }

    clusterCount -= 1;

    // recalculate updated clusters
    for (int clstrIdx : clusterChanged) {
      quadratics[clstrIdx] = clusterQuadratic(clstrIdx, labels, emb,
                                              NORMALIZE_QUADRATICS);
    }
  }

  quadratics.resize(groupCount);
}

Mat2D normalizeCoords(const Mat2D &coords, const bool NORMALIZE_DATA) {
  if (NORMALIZE_DATA) {
    const int sampleCount = coords.cols;

    Mat2D coordsHm;
    cv::vconcat(coords, Mat2D::ones(1, sampleCount), coordsHm);

    return Cholesky((coordsHm * coordsHm.t() / sampleCount).inv()) * coordsHm;
  } else {
    return coords.clone();
  }
}

Mat2D getJointData(const Mat2D &img1, const Mat2D &img2) {
  const int sampleCount = img1.cols;
  Mat2D jointImageData(dimensionCount, sampleCount);

  img1.rowRange(0, 2).copyTo(jointImageData.rowRange(0, 2));
  img2.rowRange(0, 2).copyTo(jointImageData.rowRange(2, 4));
  jointImageData.row(4) = Mat2D::ones(1, sampleCount);

  return jointImageData;
}

int ransacIter(const int sampleCount,
               const Mat2D &img1,
               const EmbValT FThreshold,
               const Mat2D &normedVector2,
               const Mat2D &kronImg1Img2,
               const std::vector<int> &groupDataIndices,
               Mat2D *F,
               IndexMat2D *currentConsensusIndices) {
  // pick first FSampleSize indices
  std::vector<int> hypIndices(groupDataIndices.begin(),
                              groupDataIndices.begin() + FSampleSize);

  // RANSAC an epipolar model
  Mat2D A = filterIdx2(kronImg1Img2, hypIndices).t();

  Mat2D V = SVD_V(A);

  // Step 2: Compute a consensus among all group samples
  Mat2D Fstacked = V.col(V.cols-1).clone();
  *F = Fstacked.reshape(0, 3).t();
  Mat2D normedVector1 = (*F) * filterIdx2(img1, groupDataIndices);

  Mat2D normedVector1sq, sums;
  cv::pow(normedVector1, 2, normedVector1sq);
  reduce(normedVector1sq, sums, 0, cv::REDUCE_SUM);
  cv::pow(sums, 0.5, sums);

  for (int i = 0; i < normedVector1.cols; i++) {
    normedVector1.col(i) /= sums(i);
  }

  Mat2D tprod;
  multiply(filterIdx2(normedVector2, groupDataIndices), normedVector1, tprod);

  Mat2D consensus;
  reduce(tprod, consensus, 0, cv::REDUCE_SUM);

  IndexMat2D temp = (cv::abs(consensus) < FThreshold)/255;

  int currentConsensus = 0;
  *currentConsensusIndices = IndexMat2D::zeros(1, sampleCount);

  for (int i = 0; i < static_cast<int>(groupDataIndices.size()); i++) {
    currentConsensus += temp(i);
    (*currentConsensusIndices)(groupDataIndices[i]) = temp(i);
  }

  return currentConsensus;
}

Mat2D constructSystemToSolveF(const Mat2D &img1, const Mat2D &img2) {
  const int sampleCount = img1.cols;
  Mat2D kronImg1Img2 = Mat2D::zeros(9, sampleCount);

  cv::multiply(img1.row(0), img2.row(0), kronImg1Img2.row(0));
  cv::multiply(img1.row(0), img2.row(1), kronImg1Img2.row(1));
  img1.row(0).copyTo(kronImg1Img2.row(2));
  cv::multiply(img1.row(1), img2.row(0), kronImg1Img2.row(3));
  cv::multiply(img1.row(1), img2.row(1), kronImg1Img2.row(4));
  img1.row(1).copyTo(kronImg1Img2.row(5));
  img2.row(0).copyTo(kronImg1Img2.row(6));
  img2.row(1).copyTo(kronImg1Img2.row(7));

  // won't work if the coordinates were not normalized (no 3rd row)
  // todo: fix here, contact authors about possible bug in Matlab code
  img2.row(2).copyTo(kronImg1Img2.row(8));

  return kronImg1Img2;
}

Mat3D recoverQuadratics(const int clusterCount,
                        const IndexMat2D &labels,
                        const EmbeddingData &emb,
                        const bool NORMALIZE_QUADRATICS) {
  Mat3D quads = Mat3D_zeros(clusterCount, dimensionCount, dimensionCount);

  for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
    quads[clusterIndex] = clusterQuadratic(clusterIndex, labels, emb,
                                           NORMALIZE_QUADRATICS);
  }

  return quads;
}

typedef std::tuple<IndexMat2D, IndexMat2D> LabelsPair;

struct Polynomial {
  Mat2D coeffs;
  Mat2D normals;
};

struct SegmentationInliers {
  Mat2D jointData;
  Polynomial polynomial;
  EmbeddingData embedding;
  std::vector<int> indices;
  EmbValT outlierPercentage;
};


LabelsPair segmentContractions(const SegmentationInliers &inliers,
                               const unsigned int groupCount,
                               const RAS_params params) {
  // Step 6: Compute Tangent Spaces and Mutual Contractions. Use Mutual
  // Contractions as a similarity metric for spectral clustering. This
  // particular algorithm tries to oversegment the sample by using a
  // conservative tolerance.
  LabelsPair out;
  IndexMat2D &trimmedLabels = std::get<0>(out);
  IndexMat2D &trimmedAllLabels = std::get<1>(out);

  const int sampleCount = inliers.jointData.cols;
  const int angleCount = static_cast<int>(params.angleTolerance.size());

  trimmedAllLabels = -1*IndexMat2D::ones(angleCount, sampleCount);

  Mat2D t1, t2, t3, t4;
  t1 = inliers.polynomial.coeffs.t() * inliers.embedding.getV();
  cv::pow(t1, 2., t2);
  cv::pow(inliers.polynomial.normals, 2., t3);
  reduce(t3, t4, 0, cv::REDUCE_SUM);

  IndexedVec normalLen(static_cast<size_t>(t2.cols));

  for (int i = 0; i < t2.cols; i++) {
    normalLen[i] = IndexedVal(t2(i)/t4(i), i);
  }

  sortIndexedVec(&normalLen);

  std::vector<int> sortedIndices(static_cast<size_t>(t2.cols));
  for (int i = 0; i < t2.cols; i++) {
    sortedIndices[i] = normalLen[i].second;
  }

  const Mat3D polynomialHessians = polyHessian(inliers.polynomial.coeffs,
                                               inliers.embedding.getH());

  const Mat2D sortedData = filterIdx2(inliers.jointData, sortedIndices);
  const Mat2D sortedNormals = filterIdx2(inliers.polynomial.normals,
                                         sortedIndices);
  const Mat3D sortedHessians = filterIdx3(polynomialHessians, sortedIndices);

  const EmbeddingData emb = perspective_embedding(sortedData, 1);

  for (int angleIndex = 0; angleIndex < angleCount; angleIndex++) {
    trimmedLabels = -1*IndexMat2D::ones(1, sampleCount);

    int clusterCount = 0;
    trimmedLabels(0) = 0;

    IndexMat2D clusterPrototype = IndexMat2D::zeros(1, sampleCount);
    clusterPrototype(0) = 0;

    EmbValT cosTolerance = cos(params.angleTolerance[angleIndex]);

    for (int sampleIdx = 1; sampleIdx < sampleCount; sampleIdx++) {
      for (int clusterIdx = 0; clusterIdx < clusterCount; clusterIdx++) {
        std::vector<int> indices{sampleIdx, clusterPrototype(clusterIdx)};

        Mat2D U = SVD_U(filterIdx2(sortedNormals, indices));

        Mat2D T = U.colRange(2, U.cols);

        Mat2D C1 = T.t() * sliceIdx3(sortedHessians, sampleIdx) * T;

        Mat2D C2 = T.t() * sliceIdx3(sortedHessians,
                                     clusterPrototype(clusterIdx)) * T;

        auto C1_ = C1.reshape(0, 1);
        auto C2_ = C2.reshape(0, C2.rows * C2.cols);

        Mat2D Cp = cv::abs(C1_ / norm(C1_) * C2_ / norm(C2_));

        if (Cp(0) >= cosTolerance) {
          trimmedLabels(sampleIdx) = clusterIdx;
          break;
        }
      }

      if (trimmedLabels(sampleIdx) == -1) {
        clusterCount = clusterCount + 1;
        trimmedLabels(sampleIdx) = clusterCount;
        clusterPrototype(clusterCount) = sampleIdx;
      }
    }

    clusterCount++;

    ////////////////////////////////////////////////////////////////////////////
    // Step 7: Recover quadratic matrices for each cluster.

    if (clusterCount < static_cast<int>(groupCount)) {
      std::runtime_error("Too few motion models found. "
                         "Please choose a smaller angleTolerance");
    }

    Mat3D quadratics = recoverQuadratics(clusterCount,
                                         trimmedLabels,
                                         emb,
                                         params.NORMALIZE_QUADRATICS);

    ////////////////////////////////////////////////////////////////////////////
    // Step 8: Kill off smaller clusters one at a time, reassigning samples
    // to remaining clusters with minimal algebraic distance.

    mergeClusters(groupCount,
                  params.NORMALIZE_QUADRATICS,
                  sortedData,
                  emb,
                  &trimmedLabels,
                  &quadratics);

    auto labelsCopy = trimmedLabels.clone();
    for (int i = 0; i < sampleCount; i++) {
      trimmedLabels(sortedIndices[i]) = labelsCopy(i);
    }

    trimmedLabels.copyTo(trimmedAllLabels.row(angleIndex));
  }

  if (params.angleTolerance.size() != 1) {
    // todo:
    // std::cout << trimmedLabels+1 << std::endl;
    // trimmedLabels = aggregate_labels(trimmedAllLabels, groupCount);
    // std::cout << trimmedLabels+1 << std::endl;
  }

  return out;
}

bool checkThreshold(const Polynomial &inlierPoly,
                    const Mat2D &inlierVeronese,
                    const RAS_params &params) {
  EmbValT maxDistance = 0;

  for (int i = 0; i < inlierVeronese.cols; i++) {
    Mat2D prod = inlierPoly.coeffs.t() * inlierVeronese.col(i);
    EmbValT currDistance = fabs(prod(0)) / norm(inlierPoly.normals.col(i));

    if (currDistance > maxDistance) {
      maxDistance = currDistance;
    }

    if (currDistance > params.boundaryThreshold && params.DEBUG <= 1) {
      break;
    }
  }

  // check if percentage below the boundary has been found
  return (params.DEBUG <= 1) && (maxDistance <= params.boundaryThreshold);
}

IndexedVec computeInfluence(const EmbeddingData &embedding,
                            const RAS_params &params) {
  const int sampleCount = embedding.getV().cols;

  IndexedVec influence(static_cast<size_t>(sampleCount));

  if (params.INFLUENCE_METHOD == InfluenceMethod::SAMPLE) {
    // polynomial.coeffs
    Mat2D tempCoeffs = find_polynomials(embedding, params.FITTING_METHOD);

    // Reject outliers by the sample influence function
    for (int i = 0; i < sampleCount; i++) {
      // compute the leave-one-out influence
      Mat2D U = find_polynomials(embedding, params.FITTING_METHOD, i);

      // polynomial.coeffs
      influence[i] = IndexedVal(subspace_angle(tempCoeffs, U), i);
    }
  } else {
    // todo
  }

  // Finally, reject outliers based on influenceValue
  sortIndexedVec(&influence);

  return influence;
}

SegmentationInliers estimateInliers(const Mat2D &jointImageData,
                                    const unsigned int groupCount,
                                    const RAS_params &params) {
  SegmentationInliers inliers;

  //////////////////////////////////////////////////////////////////////////////
  // Step 2: apply perspective veronese map to data
  const EmbeddingData emb = perspective_embedding(jointImageData, groupCount);

  //////////////////////////////////////////////////////////////////////////////
  // Step 3: Use influence function to perform robust pca
  const int sampleCount = jointImageData.cols;

  int inlierSampleCount = sampleCount;
  inliers.embedding = emb;

  IndexedVec influence(static_cast<size_t>(sampleCount));

  if (params.REJECT_KNOWN_OUTLIERS || params.REJECT_UNKNOWN_OUTLIERS) {
    influence = computeInfluence(emb, params);
  } else {
    // todo
  }

  // REJECT_UNKNOWN_OUTLIERS
  EmbValT outlierStep = (sampleCount < 100) ?
                        round(1.0 / sampleCount * 100) / 100 : 0.01;

  int outlierIndex = 0;
  inliers.indices = std::vector<int>(influence.size());

  for (inliers.outlierPercentage = params.minOutlierPercentage;
       inliers.outlierPercentage <= params.maxOutlierPercentage;
       inliers.outlierPercentage += outlierStep) {
    outlierIndex += 1;
    if (params.DEBUG > 0) {
      printf("Testing outlierPercentage %d",
             static_cast<int>(floor(100 * inliers.outlierPercentage)));
    }

    if (params.REJECT_UNKNOWN_OUTLIERS || params.REJECT_KNOWN_OUTLIERS) {
      inlierSampleCount = static_cast<int>(floor(sampleCount *
          (1 - inliers.outlierPercentage)));

      for (int i = 0; i < inlierSampleCount; i++) {
        inliers.indices[i] = influence[inlierSampleCount - 1 - i].second;
      }

      inliers.indices.resize(static_cast<size_t>(inlierSampleCount));
      inliers.embedding = EmbeddingData(emb, inliers.indices);
    }

    inliers.polynomial.coeffs = find_polynomials(inliers.embedding,
                                                 params.FITTING_METHOD);

    ////////////////////////////////////////////////////////////////////////////
    // Step 4: Compute Derivatives and Hessians for the fitting polynomial.
    inliers.polynomial.normals = Mat2D::zeros(dimensionCount,
                                              inlierSampleCount);

    for (int eIdx = 0; eIdx < dimensionCount; eIdx++) {
      inliers.polynomial.normals.row(eIdx) =
      inliers.polynomial.coeffs.t() * sliceIdx2(inliers.embedding.getD(), eIdx);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Step 5: Estimate rejection percentage via Sampson distance
    if (params.REJECT_UNKNOWN_OUTLIERS) {
      if (checkThreshold(inliers.polynomial,
                         inliers.embedding.getV(),
                         params)) {
        break;
      }
    }
  }

  if (params.REJECT_UNKNOWN_OUTLIERS &&
      inliers.outlierPercentage >= params.maxOutlierPercentage) {
    std::cout << "The RHQA function did not find an mixed motion model "
        "within the boundary threshold." << std::endl;
  }

  inliers.jointData = filterIdx2(jointImageData, inliers.indices);

  return inliers;
}

int checkDegenerateF(const Mat2D &img1,
                     const Mat2D &img2,
                     const Mat2D &normedVector2,
                     const std::vector<int> &maxFGroupDataIndices,
                     const IndexMat2D &maxFConsensusIndices,
                     IndexMat2D *maxHConsensusIndices,
                     Mat2D *bestH) {
  const int sampleCount = img1.cols;

  int maxHConsensus = 0;

  IndexMat2D allCombinations = binomMat - 1;
  for (int permutationIndex = 0; permutationIndex < allCombinations.rows;
       permutationIndex++) {
    // Step 1: Recover an H matrix
    auto indicesMat = allCombinations.row(permutationIndex);
    std::vector<int> currentDataIndices(static_cast<size_t>(indicesMat.cols));

    for (int i = 0; i < indicesMat.cols; i++) {
      currentDataIndices[i] =
              maxFGroupDataIndices[allCombinations(permutationIndex, i)];
    }

    Mat2D A = Mat2D::zeros(3 * HSampleSize, 9);

    for (int sampleIndex = 0; sampleIndex < HSampleSize; sampleIndex++) {
      auto t1 = img1.col(currentDataIndices[sampleIndex]);
      auto t2 = img2.col(currentDataIndices[sampleIndex]);
      Mat2D t3 = kron(t1, hat(t2)).t();
      t3.copyTo(A(cv::Rect(0, sampleIndex * 3, t3.cols, t3.rows)));
    }

    Mat2D V = SVD_V(A);

    // Step 2: Compute a consensus among all group samples
    Mat2D Hstacked = V.col(V.cols - 1).clone();
    Mat2D H = Hstacked.reshape(0, 3).t();

    // todo: look more carefully - why would someone want to normalize
    // scale and sign of homography matrix, which is defined up to scale?

    int currentConsensus = 0;
    IndexMat2D currentConsensusIndices = IndexMat2D::zeros(1, sampleCount);

    for (int sampleIndex = 0; sampleIndex < sampleCount;
         sampleIndex++) {
      if (maxFConsensusIndices(sampleIndex)) {
        Mat2D normedVector1 = H * img1.col(sampleIndex);
        normedVector1 /= norm(normedVector1);


        Mat2D tempm = normedVector2.col(sampleIndex).t() * normedVector1;

        EmbValT acosVal = 0;
        if (fabs(tempm(0)) < 1) {
          acosVal = acos(fabs(tempm(0)));
        }

        if (acosVal < HAngleThreshold) {
          // Two vectors are parallel, hit one consensus
          currentConsensus += 1;
          currentConsensusIndices(0, sampleIndex) = 1;
        }
      }
    }

    if (currentConsensus > maxHConsensus) {
      maxHConsensus = currentConsensus;
      *maxHConsensusIndices = currentConsensusIndices.clone();
      *bestH = H.clone();
    }
  }

  return maxHConsensus;
}

IndexMat2D postRansac(const Mat2D &img1,
                      const Mat2D &img2,
                      const RAS_params &params,
                      const IndexMat2D &labels,
                      IndexMat2D *motionModelsP) {
  IndexMat2D &motionModels = *motionModelsP;

  const int sampleCount = img1.cols;
  const int groupCount = motionModels.cols;

  const int RANSACIteration = 10;

  IndexMat2D currentLabels = -2 * IndexMat2D::ones(1, sampleCount);

  auto normedVector2 = img2.clone();
  for (int sampleIdx = 0; sampleIdx < sampleCount; sampleIdx++) {
    normedVector2.col(sampleIdx) /= norm(normedVector2.col(sampleIdx));
  }

  const Mat2D kronImg1Img2 = constructSystemToSolveF(img1, img2);

  Mat3D bestF = Mat3D_zeros(groupCount, 3, 3);
  Mat3D bestH = Mat3D_zeros(groupCount, 3, 3);

  std::vector<int> maxFGroupDataIndices;

  IndexMat2D isGroupOutliers = IndexMat2D::zeros(groupCount, sampleCount);

  for (int grpIdx = 0; grpIdx < groupCount; grpIdx++) {
    std::vector<int> groupDataIndices =
            filterIndices(labels, [grpIdx](int val){return val == grpIdx;});

    int maxFConsensus = 0;
    int maxHConsensus = 0;
    IndexMat2D maxFConsensusIndices;
    IndexMat2D maxHConsensusIndices;

    int groupSize = static_cast<int>(groupDataIndices.size());

    // Matlab lacks check if there are enough points in group for F hypothesis
    // todo: contact authors about possible bug
    if (groupSize < FSampleSize) {
      std::cout << "not enough points (" << groupSize << ") in group #" <<
      grpIdx << std::endl;
      // todo: add more handling logic (mark points as outliers, whatever...)
      continue;
    }

    for (int iterationIndex = 0; iterationIndex < RANSACIteration;
        iterationIndex++) {
      // if(iterationIndex+FSampleSize >= groupDataIndices.size()) break;
      // Sample 8 points
      // std::random_shuffle(groupDataIndices.begin(), groupDataIndices.end());

      Mat2D F;
      IndexMat2D currentConsensusIndices;
      int currentConsensus = ransacIter(sampleCount,
                                        img1,
                                        FThreshold,
                                        normedVector2,
                                        kronImg1Img2,
                                        groupDataIndices,
                                        &F,
                                        &currentConsensusIndices);

      if (currentConsensus > maxFConsensus) {
        maxFConsensus = currentConsensus;
        maxFConsensusIndices = currentConsensusIndices.clone();

        maxFGroupDataIndices = std::vector<int>(groupDataIndices.begin(),
                                                groupDataIndices.begin() +
                                                    FSampleSize);

        bestF[grpIdx] = F;
      }

      break;
    }

    if (maxFConsensus > 0) {
      // Further verify if the F is a degenerate H relation
      maxHConsensus = checkDegenerateF(img1,
                                       img2,
                                       normedVector2,
                                       maxFGroupDataIndices,
                                       maxFConsensusIndices,
                                       &maxHConsensusIndices,
                                       &(bestH[grpIdx]));
    }

    // Finally assign labels and mode
    if (maxFConsensus == 0) {
      // All outliers
      motionModels(grpIdx) = -1;

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == grpIdx) {
          isGroupOutliers(grpIdx, i) = 1;
        }
      }
    } else if (maxHConsensus > maxFConsensus/3*2) {
      motionModels(grpIdx) = 2;

      for (int i = 0; i < maxHConsensusIndices.cols; i++) {
        if (maxHConsensusIndices(i))
          currentLabels(i) = grpIdx;
      }

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == grpIdx && !(maxHConsensusIndices(i))) {
          isGroupOutliers(grpIdx, i) = 1;
        }
      }
    } else {
      // Epipolar consensus is high
      motionModels(grpIdx) = 1;

      for (int i = 0; i < maxFConsensusIndices.cols; i++) {
        if (maxFConsensusIndices(i))
          currentLabels(i) = grpIdx;
      }

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == grpIdx && !(maxFConsensusIndices(i))) {
          isGroupOutliers(grpIdx, i) = 1;
        }
      }
    }
  }

  if (params.RETEST_OUTLIERS) {
    // todo
    // dont forget about changes made to bestF and bestH indexing
  }

  std::cout << isGroupOutliers << std::endl;

  return currentLabels.clone();
}

// put inlier labels in the output according to the inlierIndices, set -1
// label for the outliers
IndexMat2D recomposeLabels(const int sampleCount,
                           const IndexMat2D &inlierLabels,
                           const std::vector<int> &inlierIndices,
                           const RAS_params &params) {
  if (params.REJECT_UNKNOWN_OUTLIERS || params.REJECT_KNOWN_OUTLIERS) {
    IndexMat2D labels = -1*IndexMat2D::ones(inlierLabels.rows, sampleCount);

    for (int i = 0; i < static_cast<int>(inlierIndices.size()); i++) {
      inlierLabels.col(i).copyTo(labels.col(inlierIndices[i]));
    }

    return labels;
  } else {
    return inlierLabels.clone();
  }
}

Mat2D robust_algebraic_segmentation(const Mat2D &img1Unnorm,
                                    const Mat2D &img2Unnorm,
                                    const unsigned int groupCount,
                                    const RAS_params &params) {
  //////////////////////////////////////////////////////////////////////////////
  // Step 1: map coordinates to joint image space.
  const Mat2D img1 = normalizeCoords(img1Unnorm, params.NORMALIZE_DATA);
  const Mat2D img2 = normalizeCoords(img2Unnorm, params.NORMALIZE_DATA);

  const Mat2D jointImageData = getJointData(img1, img2);

  //////////////////////////////////////////////////////////////////////////////
  // Steps 2-5
  const SegmentationInliers inliers = estimateInliers(jointImageData,
                                                      groupCount,
                                                      params);

  //////////////////////////////////////////////////////////////////////////////
  // Steps 6-8
  IndexMat2D inlierLabels, inlierAllLabels;

  std::tie(inlierLabels, inlierAllLabels) = segmentContractions(inliers,
                                                                groupCount,
                                                                params);

  const int sampleCount = jointImageData.cols;

  IndexMat2D labels = recomposeLabels(sampleCount,
                                      inlierLabels,
                                      inliers.indices,
                                      params),
      allLabels = recomposeLabels(sampleCount,
                                  inlierAllLabels,
                                  inliers.indices,
                                  params);

  //////////////////////////////////////////////////////////////////////////////
  // Step 9: Post-refinement using RANSAC
  // For each group, run RANSAC to re-estimate a motion model.
  // After the refinement, resegment the data based on the motion model.

  IndexMat2D motionModels = IndexMat2D::zeros(1, groupCount);

  if (params.POST_RANSAC) {
    labels = postRansac(img1, img2, params, labels, &motionModels);
  }

  std::cout << labels+1 << std::endl;
  std::cout << motionModels << std::endl;

  return Mat2D();
}
