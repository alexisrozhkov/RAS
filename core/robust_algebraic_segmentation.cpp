// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <core/perspective_embedding.h>
#include <core/utils/cholesky.h>
#include <core/utils/arma_svd.h>
#include <core/utils/subspace_angle.h>

#include <vector>
#include <limits>
#include <utility>
#include <iostream>
#include <algorithm>


typedef std::pair<EmbValT, int> InflVal;

struct sort_pred {
  bool operator()(const InflVal &left, const InflVal &right) {
    return left.first < right.first;
  }
};

static const int HSampleSize = 4;
static const int FSampleSize = 8;
static const int dimensionCount = 5;

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


Mat3D polyHessian(const int samplesCount,
                  const Mat2D &coeffs,
                  const Mat4D &hessian) {
  Mat3D hpn = Mat3D_zeros(dimensionCount, dimensionCount, samplesCount);

  for (int idx1 = 0; idx1 < dimensionCount; idx1++) {
    for (int idx2 = idx1; idx2 < dimensionCount; idx2++) {
      hpn[idx1].row(idx2) = coeffs.t() * sliceIdx23(hessian, idx1, idx2);

      if (idx1 != idx2) {
        hpn[idx1].row(idx2).copyTo(hpn[idx2].row(idx1));
      }
    }
  }

  return hpn;
}

template <class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator copy_idx_if (InputIterator first, InputIterator last,
                        OutputIterator result, UnaryPredicate pred)
{
  InputIterator start = first;

  while (first!=last) {
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

Mat2D clusterQuadratic(const int clusterIdx,
                       const IndexMat2D &labels,
                       const Mat2D &embeddedData,
                       const Mat4D &HembeddedData,
                       const bool NORMALIZE_QUADRATICS) {
  std::vector<int> currIndices =
      filterIndices(labels, [clusterIdx](int val){return val == clusterIdx;});

  int clusterSize = static_cast<int>(currIndices.size());
  auto clusterData = filterIdx2(embeddedData, currIndices);
  auto clusterHessian = filterIdx4(HembeddedData, currIndices);

  Mat2D U;
  armaSVD_U(clusterData, &U);

  Mat3D hpn = polyHessian(clusterSize,
                          U.col(U.cols - 1),
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

void step8(const unsigned int groupCount,
           const bool NORMALIZE_QUADRATICS,
           const Mat2D &jointImageData,
           const Mat2D &embeddedData,
           const Mat4D &HembeddedData,
           IndexMat2D &labels,
           Mat3D &quadratics) {
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
        // swap smallestClust and clusterCount-1 labels
        if (labels(i) == smallestClust) {
          labels(i) = clusterCount-1;
        } else if (labels(i) == clusterCount-1) {
          labels(i) = smallestClust;
        }
      }

      // swap smallestClust and clusterCount-1 quadratics
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
      quadratics[clstrIdx] = clusterQuadratic(clstrIdx,
                                              labels,
                                              embeddedData,
                                              HembeddedData,
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

  Mat2D V;
  armaSVD_V(A, &V);

  // Step 2: Compute a consensus among all group samples
  Mat2D Fstacked = V.col(V.cols-1).clone();
  *F = Fstacked.reshape(0, 3).t();
  Mat2D normedVector1 = (*F) * filterIdx2(img1, groupDataIndices);

  Mat2D normedVector1sq, sums;
  cv::pow(normedVector1, 2, normedVector1sq);
  reduce(normedVector1sq, sums, 0, cv::REDUCE_SUM);
  cv::pow(sums, 0.5, sums);

  for(int i = 0; i < normedVector1.cols; i++) {
    normedVector1.col(i) /= sums(i);
  }

  Mat2D tprod;
  multiply(filterIdx2(normedVector2, groupDataIndices), normedVector1, tprod);

  Mat2D consensus;
  reduce(tprod, consensus, 0, cv::REDUCE_SUM);

  IndexMat2D temp = (cv::abs(consensus) < FThreshold)/255;

  int currentConsensus = 0;
  *currentConsensusIndices = IndexMat2D::zeros(1, sampleCount);

  for(int i = 0; i < static_cast<int>(groupDataIndices.size()); i++) {
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
                        const Mat2D &embeddedData,
                        const Mat4D &HembeddedData,
                        const bool NORMALIZE_QUADRATICS) {
  Mat3D quads = Mat3D_zeros(clusterCount, dimensionCount, dimensionCount);

  for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
    quads[clusterIndex] = clusterQuadratic(clusterIndex,
                                           labels,
                                           embeddedData,
                                           HembeddedData,
                                           NORMALIZE_QUADRATICS);
  }

  return quads;
}

void step6(const unsigned int groupCount,
           const std::vector<EmbValT> &angleTolerance,
           const bool NORMALIZE_QUADRATICS,
           Mat2D &jointImageData,
           const Mat2D &veroneseData,
           Mat4D &veroneseHessian,
           const Mat2D &polynomialCoefficients,
           Mat2D &polynomialNormals,
           IndexMat2D &labels,
           IndexMat2D &allLabels) {
  const int sampleCount = jointImageData.cols;
  const int angleCount = static_cast<int>(angleTolerance.size());

  allLabels = -1*IndexMat2D::ones(angleCount, sampleCount);

  Mat2D t1, t2, t3, t4;
  t1 = polynomialCoefficients.t() * veroneseData;
  cv::pow(t1, 2., t2);
  cv::pow(polynomialNormals, 2., t3);
  reduce(t3, t4, 0, cv::REDUCE_SUM);

  std::vector<InflVal> normalLen(static_cast<size_t>(t2.cols));

  for (int i = 0; i < t2.cols; i++) {
    normalLen[i] = InflVal(t2(i)/t4(i), i);
  }

  sort(normalLen.begin(), normalLen.end(), sort_pred());

  std::vector<int> sortedIndices(static_cast<size_t>(t2.cols));
  for (int i = 0; i < t2.cols; i++) {
    sortedIndices[i] = normalLen[i].second;
  }

  Mat3D polynomialHessians = polyHessian(sampleCount,
                                         polynomialCoefficients,
                                         veroneseHessian);

  jointImageData = filterIdx2(jointImageData, sortedIndices);
  polynomialNormals = filterIdx2(polynomialNormals, sortedIndices);
  polynomialHessians = filterIdx3(polynomialHessians, sortedIndices);

  auto emb = perspective_embedding(jointImageData, 1, false);
  auto embeddedData = emb.getV().back();
  auto DembeddedData = emb.getD().back();
  auto HembeddedData = emb.getH().back();

  for (int angleIndex = 0; angleIndex < angleCount; angleIndex++) {
    labels = -1*IndexMat2D::ones(1, sampleCount);

    int clusterCount = 0;
    labels(0) = 0;

    IndexMat2D clusterPrototype = IndexMat2D::zeros(1, sampleCount);
    clusterPrototype(0) = 0;

    EmbValT cosTolerance = cos(angleTolerance[angleIndex]);

    for (int sampleIdx = 1; sampleIdx < sampleCount; sampleIdx++) {
      for (int clusterIdx = 0; clusterIdx < clusterCount; clusterIdx++) {
        std::vector<int> indices{sampleIdx, clusterPrototype(clusterIdx)};

        Mat2D U;
        armaSVD_U(filterIdx2(polynomialNormals, indices), &U);

        Mat2D T = U.colRange(2, U.cols);

        Mat2D C1 = T.t() * sliceIdx3(polynomialHessians, sampleIdx) * T;

        Mat2D C2 = T.t() * sliceIdx3(polynomialHessians,
                                     clusterPrototype(clusterIdx)) * T;

        auto C1_ = C1.reshape(0, 1);
        auto C2_ = C2.reshape(0, C2.rows * C2.cols);

        Mat2D Cp = cv::abs(C1_ / norm(C1_) * C2_ / norm(C2_));

        if (Cp(0) >= cosTolerance) {
          labels(sampleIdx) = clusterIdx;
          break;
        }
      }

      if (labels(sampleIdx) == -1) {
        clusterCount = clusterCount + 1;
        labels(sampleIdx) = clusterCount;
        clusterPrototype(clusterCount) = sampleIdx;
      }
    }

    clusterCount++;

    ////////////////////////////////////////////////////////////////////////////
    // Step 7: Recover quadratic matrices for each cluster.

    if (clusterCount < static_cast<int>(groupCount)) {
      std::runtime_error("Too few motion models found. "\
                         "Please choose a smaller angleTolerance");
    }

    Mat3D quadratics = recoverQuadratics(clusterCount,
                                         labels,
                                         embeddedData,
                                         HembeddedData,
                                         NORMALIZE_QUADRATICS);

    ////////////////////////////////////////////////////////////////////////////
    // Step 8: Kill off smaller clusters one at a time, reassigning samples
    // to remaining clusters with minimal algebraic distance.

    step8(groupCount,
          NORMALIZE_QUADRATICS,
          jointImageData,
          embeddedData,
          HembeddedData,
          labels,
          quadratics);

    auto labelsCopy = labels.clone();
    for (int i = 0; i < sampleCount; i++) {
      labels(sortedIndices[i]) = labelsCopy(i);
    }

    labels.copyTo(allLabels.row(angleIndex));
  }
}

void step3(const RAS_params &params,
           const Embedding &embedding,
           const Mat2D &jointImageData,
           Mat2D &polynomialCoefficients,
           Mat2D &polynomialNormals,
           std::vector<int> &inlierIndices,
           EmbValT &outlierPercentage,
           int &trimmedSampleCount,
           Mat2D &trimmedData,
           Mat2D &trimmedVeronese,
           Mat3D &trimmedDerivative,
           Mat4D &trimmedHessian) {
  const auto untrimmedVeronese = embedding.getV().back();
  const auto untrimmedDerivative = embedding.getD().back();
  const auto untrimmedHessian = embedding.getH().back();

  const int untrimmedSampleCount = jointImageData.cols;

  trimmedSampleCount = untrimmedSampleCount;
  trimmedData = jointImageData.clone();
  trimmedVeronese = untrimmedVeronese.clone();
  trimmedDerivative = untrimmedDerivative;
  trimmedHessian = untrimmedHessian;

  std::vector<InflVal> sortedIndex(static_cast<size_t>(untrimmedSampleCount));
  std::vector<InflVal> influenceValues(static_cast<size_t>(untrimmedSampleCount));

  if (params.REJECT_KNOWN_OUTLIERS || params.REJECT_UNKNOWN_OUTLIERS) {
    if (params.INFLUENCE_METHOD == InfluenceMethod::SAMPLE) {
      polynomialCoefficients = find_polynomials(untrimmedVeronese,
                                                untrimmedDerivative,
                                                params.FITTING_METHOD,
                                                1);

      // Reject outliers by the sample influence function
      for (int sIdx = 0; sIdx < untrimmedSampleCount; sIdx++) {
        // compute the leave-one-out influence
        auto U = find_polynomials(untrimmedVeronese, untrimmedDerivative,
                                  params.FITTING_METHOD, 1, sIdx);

        influenceValues[sIdx] =
            InflVal(subspace_angle(polynomialCoefficients, U), sIdx);
      }
    } else {
      // todo
    }

    sortedIndex = influenceValues;

    // Finally, reject outliers based on influenceValue
    sort(sortedIndex.begin(),
         sortedIndex.end(),
         sort_pred());
  }

  // REJECT_UNKNOWN_OUTLIERS
  EmbValT outlierStep = (untrimmedSampleCount < 100) ?
                        round(1.0 / untrimmedSampleCount * 100) / 100 : 0.01;

  size_t numDistanceStackElem = static_cast<size_t>(
      (params.maxOutlierPercentage - params.minOutlierPercentage)/outlierStep) + 1;

  std::vector<EmbValT> distanceStack(numDistanceStackElem);

  int outlierIndex = 0;
  inlierIndices = std::vector<int>(sortedIndex.size());

  for (outlierPercentage = params.minOutlierPercentage;
       outlierPercentage <= params.maxOutlierPercentage;
       outlierPercentage += outlierStep) {
    outlierIndex += 1;
    if (params.DEBUG > 0) {
      printf("Testing outlierPercentage %d",
             static_cast<int>(floor(100 * outlierPercentage)));
    }

    if (params.REJECT_UNKNOWN_OUTLIERS || params.REJECT_KNOWN_OUTLIERS) {
      trimmedSampleCount = static_cast<int>(floor(untrimmedSampleCount *
          (1 - outlierPercentage)));

      for (int i = 0; i < trimmedSampleCount; i++) {
        inlierIndices[i] = sortedIndex[trimmedSampleCount - 1 - i].second;
      }

      inlierIndices.resize(static_cast<size_t>(trimmedSampleCount));

      trimmedData = filterIdx2(jointImageData, inlierIndices);
      trimmedVeronese = filterIdx2(untrimmedVeronese, inlierIndices);
      trimmedDerivative = filterIdx3(untrimmedDerivative, inlierIndices);
      trimmedHessian = filterIdx4(untrimmedHessian, inlierIndices);
    }

    polynomialCoefficients = find_polynomials(trimmedVeronese,
                                              trimmedDerivative,
                                              params.FITTING_METHOD, 1);

    ////////////////////////////////////////////////////////////////////////////
    // Step 4: Compute Derivatives and Hessians for the fitting polynomial.
    polynomialNormals = Mat2D::zeros(dimensionCount, trimmedSampleCount);

    for (int eIdx = 0; eIdx < dimensionCount; eIdx++) {
      polynomialNormals.row(eIdx) =
          polynomialCoefficients.t() * sliceIdx2(trimmedDerivative, eIdx);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Step 5: Estimate rejection percentage via Sampson distance

    if (params.REJECT_UNKNOWN_OUTLIERS) {
      distanceStack[outlierIndex] = 0;

      for (int inlierIndex = 0; inlierIndex < trimmedSampleCount; inlierIndex++) {
        Mat2D prod = polynomialCoefficients.t() * trimmedVeronese.col(inlierIndex);
        EmbValT sampsonDistance = fabs(prod(0)) /
            norm(polynomialNormals.col(inlierIndex));

        if (sampsonDistance > distanceStack[outlierIndex]) {
          distanceStack[outlierIndex] = sampsonDistance;
        }

        if (sampsonDistance > params.boundaryThreshold && params.DEBUG <= 1) {
          break;
        }
      }

      if ((params.DEBUG <= 1) &&
          (distanceStack[outlierIndex] <= params.boundaryThreshold)) {
        // One percentage below the boundary is found
        break;
      }
    }
  }
}


Mat2D hat(const Mat2D &v) {
  return (Mat2D(3, 3) <<    0, -v(2),  v(1),
                         v(2),     0, -v(0),
                        -v(1),  v(0),    0);
}

Mat2D kron(const Mat2D &a, const Mat2D &b) {
  Mat2D out(a.rows*b.rows, a.cols*b.cols);

  for(int j = 0; j < a.rows; j++) {
    for(int i = 0; i < a.cols; i++) {
      out(cv::Rect(i*b.cols, j*b.rows, b.cols, b.rows)) = a(j, i)*b;
    }
  }

  return out;
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
  // Step 2: apply perspective veronese map to data
  const Embedding embedding = perspective_embedding(jointImageData,
                                                    groupCount, false);

  //////////////////////////////////////////////////////////////////////////////
  // Step 3: Use influence function to perform robust pcaMat2D untrimmedData;
  Mat2D polynomialCoefficients;
  Mat2D polynomialNormals;

  std::vector<int> inlierIndices;
  EmbValT outlierPercentage;

  int trimmedSampleCount;
  Mat2D trimmedData;
  Mat2D trimmedVeronese;
  Mat3D trimmedDerivative;
  Mat4D trimmedHessian;

  step3(params,
        embedding,
        jointImageData,
        polynomialCoefficients,
        polynomialNormals,
        inlierIndices,
        outlierPercentage,
        trimmedSampleCount,
        trimmedData,
        trimmedVeronese,
        trimmedDerivative,
        trimmedHessian);

  //////////////////////////////////////////////////////////////////////////////
  // Step 6: Compute Tangent Spaces and Mutual Contractions. Use Mutual
  // Contractions as a similarity metric for spectral clustering. This
  // particular algorithm tries to oversegment the sample by using a
  // conservative tolerance.

  IndexMat2D labels, allLabels;

  step6(groupCount,
        params.angleTolerance,
        params.NORMALIZE_QUADRATICS,
        trimmedData,
        trimmedVeronese,
        trimmedHessian,
        polynomialCoefficients,
        polynomialNormals,
        labels,
        allLabels);

  const int sampleCount = jointImageData.cols;
  const int angleCount = static_cast<int>(params.angleTolerance.size());

  if (angleCount != 1) {
    // todo:
    //std::cout << labels+1 << std::endl;
    // labels = aggregate_labels(allLabels, groupCount);
    //std::cout << labels+1 << std::endl;
  }

  if (params.REJECT_UNKNOWN_OUTLIERS || params.REJECT_KNOWN_OUTLIERS) {
    IndexMat2D untrimmedAllLabels = -1*IndexMat2D::ones(angleCount,
                                                        sampleCount);

    for (int i = 0; i < static_cast<int>(inlierIndices.size()); i++) {
      allLabels.col(i).copyTo(untrimmedAllLabels.col(inlierIndices[i]));
    }

    allLabels = untrimmedAllLabels;

    IndexMat2D untrimmedLabels = -1*IndexMat2D::ones(1, sampleCount);

    for (int i = 0; i < static_cast<int>(inlierIndices.size()); i++) {
      untrimmedLabels(inlierIndices[i]) = labels(i);
    }

    labels = untrimmedLabels.clone();
    trimmedData = jointImageData.clone();
    trimmedSampleCount = sampleCount;
  }

  if (params.REJECT_UNKNOWN_OUTLIERS &&
      outlierPercentage >= params.maxOutlierPercentage) {
    std::cout << "The RHQA function did not find an mixed motion model "\
                 "within the boundary threshold." << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Step 9: Post-refinement using RANSAC
  // For each group, run RANSAC to re-estimate a motion model.
  // After the refinement, resegment the data based on the motion model.

  IndexMat2D motionModels = IndexMat2D::zeros(1, groupCount);
  IndexMat2D isGroupOutliers = IndexMat2D::zeros(groupCount, trimmedSampleCount);

  if (params.POST_RANSAC) {
    const int RANSACIteration = 10;
    const EmbValT FThreshold = 0.035;  // FThreshold = 0.015;
    const EmbValT HAngleThreshold = 0.04;


    IndexMat2D currentLabels = -2*IndexMat2D::ones(1, trimmedSampleCount);

    auto normedVector2 = img2.clone();
    for (int sampleIdx = 0; sampleIdx < trimmedSampleCount; sampleIdx++) {
      normedVector2.col(sampleIdx) /= cv::norm(normedVector2.col(sampleIdx));
    }

    const Mat2D kronImg1Img2 = constructSystemToSolveF(img1, img2);

    //std::cout << kronImg1Img2 << std::endl;

    Mat3D bestF = Mat3D_zeros(groupCount, 3, 3);
    Mat3D bestH = Mat3D_zeros(groupCount, 3, 3);

    std::vector<int> maxFGroupDataIndices;

    for (int grpIdx = 0; grpIdx < static_cast<int>(groupCount); grpIdx++) {
      std::vector<int> groupDataIndices;

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == grpIdx) {
          groupDataIndices.push_back(i);
        }
      }

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

        //if(iterationIndex+FSampleSize >= groupDataIndices.size()) break;
        // Sample 8 points
        //std::random_shuffle(groupDataIndices.begin(), groupDataIndices.end());

        Mat2D F;
        IndexMat2D currentConsensusIndices;
        int currentConsensus = ransacIter(trimmedSampleCount,
                                          img1,
                                          FThreshold,
                                          normedVector2,
                                          kronImg1Img2,
                                          groupDataIndices,
                                          &F,
                                          &currentConsensusIndices);

        if (currentConsensus > maxFConsensus) {
          maxFConsensus = currentConsensus;
          maxFConsensusIndices = currentConsensusIndices;

          maxFGroupDataIndices = std::vector<int>(groupDataIndices.begin()+iterationIndex,
                                                  groupDataIndices.begin()+(iterationIndex+FSampleSize));

          bestF[grpIdx] = F;
        }

        break;
      }

      if (maxFConsensus > 0) {
        // Further verify if the F is a degenerate H relation
        IndexMat2D allCombinations = binomMat-1;
        maxHConsensus = 0;

        for (int permutationIndex=0; permutationIndex < allCombinations.rows;
          permutationIndex++) {
          // Step 1: Recover an H matrix
          auto indicesMat = allCombinations.row(permutationIndex);
          std::vector<int> currentDataIndices(indicesMat.cols);

          for(int i = 0; i < indicesMat.cols; i++) {
            currentDataIndices[i] =
                maxFGroupDataIndices[allCombinations(permutationIndex, i)];
          }

          Mat2D A = Mat2D::zeros(3*HSampleSize, 9);

          for (int sampleIndex = 0; sampleIndex < HSampleSize; sampleIndex++) {
            auto t1 = img1.col(currentDataIndices[sampleIndex]);
            auto t2 = img2.col(currentDataIndices[sampleIndex]);
            Mat2D t3 = kron(t1, hat(t2)).t();
            t3.copyTo(A(cv::Rect(0, sampleIndex*3, t3.cols, t3.rows)));
          }

          Mat2D V;
          armaSVD_V(A, &V);

          // Step 2: Compute a consensus among all group samples
          Mat2D Hstacked = V.col(V.cols-1).clone();
          Mat2D H = Hstacked.reshape(0, 3).t();

          // todo: look more carefully - why would someone want to normalize
          // sign of homography matrix, which is defined up to scale?

          /*
          Mat2D S;
          armaSVD_S(H, &S);

          H /= S(1, 1);

         // 3.2. normalize the sign of matrix H

         IndexMat2D signArray = IndexMat2D::zeros(1, HSampleSize);
          for (int sampleIndex=0; sampleIndex < HSampleSize; sampleIndex++) {
          signArray(sampleIndex) = sign(img2(:,currentDataIndices(sampleIndex)).'*H*img1(:,currentDataIndices(sampleIndex)));
          }

          if (sum(signArray)<0){
            H = -H;
          }
           */

          int currentConsensus = 0;
          IndexMat2D currentConsensusIndices = IndexMat2D::zeros(1, trimmedSampleCount);

          for (int sampleIndex = 0; sampleIndex < trimmedSampleCount;
              sampleIndex++) {
            if (maxFConsensusIndices(sampleIndex)) {
              Mat2D normedVector1 = H*img1.col(sampleIndex);
              normedVector1 /= cv::norm(normedVector1);


              Mat2D tempm = normedVector2.col(sampleIndex).t() *
                            normedVector1;

              EmbValT acosVal = 0;
              if(fabs(tempm(0)) < 1) {
                acosVal = acos(fabs(tempm(0)));
              }
              //std::cout << acosVal << std::endl << std::endl;

              if(acosVal < HAngleThreshold) {
                // Two vectors are parallel, hit one consensus
                currentConsensus += 1;
                currentConsensusIndices(0, sampleIndex) = 1;
              }
            }
          }

          if (currentConsensus > maxHConsensus) {
            maxHConsensus = currentConsensus;
            maxHConsensusIndices = currentConsensusIndices.clone();
            bestH[grpIdx] = H;
          }
        }
      }

      // Finally assign labels and mode
      if(maxFConsensus == 0) {
        // All outliers
        motionModels(grpIdx) = -1;

        for(int i = 0; i < labels.cols; i++) {
          if(labels(i) == grpIdx) {
            isGroupOutliers(grpIdx, i) = 1;
          }
        }
      } else if(maxHConsensus > maxFConsensus/3*2) {
        motionModels(grpIdx) = 2;

        for(int i = 0; i < maxHConsensusIndices.cols; i++) {
          if(maxHConsensusIndices(i))
            currentLabels(i) = grpIdx;
        }

        for(int i = 0; i < labels.cols; i++) {
          if(labels(i) == grpIdx && !(maxHConsensusIndices(i))) {
            isGroupOutliers(grpIdx, i) = 1;
          }
        }
      } else {
        // Epipolar consensus is high
        motionModels(grpIdx) = 1;

        for(int i = 0; i < maxFConsensusIndices.cols; i++) {
          if(maxFConsensusIndices(i))
            currentLabels(i) = grpIdx;
        }

        for(int i = 0; i < labels.cols; i++) {
          if(labels(i) == grpIdx && !(maxFConsensusIndices(i))) {
            isGroupOutliers(grpIdx, i) = 1;
          }
        }
      }
    }

    if (params.RETEST_OUTLIERS) {
      // todo
      // dont forget about changes made to bestF and bestH indexing
    }

    labels = currentLabels.clone();
  }

  std::cout << labels+1 << std::endl;
  std::cout << motionModels << std::endl;
  std::cout << isGroupOutliers << std::endl;

  return Mat2D();
}