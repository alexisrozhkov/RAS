// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <core/perspective_embedding.h>
#include <core/utils/cholesky.h>
#include <core/utils/jacobi_svd.h>
#include <core/utils/subspace_angle.h>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>


typedef std::pair<EmbValT, int> InflVal;

struct sort_pred {
  bool operator()(const InflVal &left, const InflVal &right) {
    return left.first < right.first;
  }
};

Mat2D mapPerspective(const Mat2D &img1,
                     const Mat2D &img2,
                     const bool NORMALIZE_DATA) {
  const int sampleCount = img1.cols;
  const int dimensionCount = 5;

  Mat2D jointImageData = Mat2D::zeros(dimensionCount, sampleCount);

  if (NORMALIZE_DATA) {
    Mat2D img1_, img2_, temp;
    cv::vconcat(img1, Mat2D::ones(1, sampleCount), img1_);
    cv::vconcat(img2, Mat2D::ones(1, sampleCount), img2_);

    temp = Cholesky((img1_ * img1_.t() / sampleCount).inv()) * img1_;
    temp.rowRange(0, 2).copyTo(jointImageData.rowRange(0, 2));

    temp = Cholesky((img2_ * img2_.t() / sampleCount).inv()) * img2_;
    temp.rowRange(0, 2).copyTo(jointImageData.rowRange(2, 4));
  } else {
    img1.copyTo(jointImageData.rowRange(0, 2));
    img2.copyTo(jointImageData.rowRange(2, 4));
  }

  jointImageData.row(4) = Mat2D::ones(1, sampleCount);

  return jointImageData;
}

EmbValT chooseMinOutlierPercentage(const EmbValT minOutlierPercentage,
                                   const EmbValT maxOutlierPercentage) {
  const EmbValT defaultVal = 0;

  if (minOutlierPercentage == NotSpecified) {  // nothing given
    return defaultVal;
  } else {  // one or two values given
    return minOutlierPercentage;
  }
}

EmbValT chooseMaxOutlierPercentage(const EmbValT minOutlierPercentage,
                                   const EmbValT maxOutlierPercentage) {
  const EmbValT defaultVal = 0.5;

  if (minOutlierPercentage == NotSpecified) {  // nothing given
    return defaultVal;
  } else if (maxOutlierPercentage == NotSpecified) {  // one value given
    return minOutlierPercentage;
  } else {
    return maxOutlierPercentage;  // range given
  }
}

bool chooseRKO(const EmbValT minOutlierPercentage,
               const EmbValT maxOutlierPercentage) {
  return (minOutlierPercentage > 0) &&
         (maxOutlierPercentage == NotSpecified);
}

bool chooseRUO(const EmbValT boundaryThreshold,
               const EmbValT minOutlierPercentage,
               const EmbValT maxOutlierPercentage) {
  return boundaryThreshold != NotSpecified ||
      ((minOutlierPercentage != NotSpecified) &&
       (maxOutlierPercentage != NotSpecified));
}

Mat3D polyHessian(const int dimCount,
                  const int samplesCount,
                  const Mat2D &coeffs,
                  const Mat4D &hessian) {
  Mat3D hpn = Mat3D_zeros(dimCount, dimCount, samplesCount);

  for (int idx1 = 0; idx1 < dimCount; idx1++) {
    for (int idx2 = idx1; idx2 < dimCount; idx2++) {
      hpn[idx1].row(idx2) = coeffs.t() * sliceIdx23(hessian, idx1, idx2);

      if (idx1 != idx2) {
        hpn[idx1].row(idx2).copyTo(hpn[idx2].row(idx1));
      }
    }
  }

  return hpn;
}

Mat2D robust_algebraic_segmentation(const Mat2D &img1,
                                    const Mat2D &img2,
                                    const unsigned int groupCount,

                                    const EmbValT boundaryThreshold_,
                                    const EmbValT minOutlierPercentage_,
                                    const EmbValT maxOutlierPercentage_,

                                    const int debug,
                                    const bool postRansac,
                                    const EmbValT angleTolerance,
                                    const FindPolyMethod fittingMethod,
                                    const InfluenceMethod influenceMethod,
                                    const bool normalizeCoordinates,
                                    const bool normalizeQuadratics,
                                    const bool retestOutliers) {
  //////////////////////////////////////////////////////////////////////////////
  // parse arguments (currently as close as possible to Matlab, should be
  // refactored later
  const int DEBUG = debug;
  const bool POST_RANSAC = postRansac;

  CV_Assert(!(angleTolerance < 0 || angleTolerance >= CV_PI / 2));

  // todo: handle unspecified value of boundaryThreshold more gracefully
  const EmbValT boundaryThreshold = (boundaryThreshold_ == NotSpecified) ? 0 :
                                    boundaryThreshold_;

  // will not complain if boundaryThreshold_ was -1, since it's a default value
  // and will be overwritten with 0
  CV_Assert(boundaryThreshold >= 0);

  const FindPolyMethod FITTING_METHOD = fittingMethod;
  const InfluenceMethod INFLUENCE_METHOD = influenceMethod;

  // todo: check if it is an error/typo in Matlab implementation
  // const bool NORMALIZE_COORDINATES = true;
  const bool NORMALIZE_DATA = normalizeCoordinates;

  const bool NORMALIZE_QUADRATICS = normalizeQuadratics;

  const EmbValT minOutlierPercentage =
      chooseMinOutlierPercentage(minOutlierPercentage_, maxOutlierPercentage_);

  const EmbValT maxOutlierPercentage =
      chooseMaxOutlierPercentage(minOutlierPercentage_, maxOutlierPercentage_);

  const bool REJECT_UNKNOWN_OUTLIERS = chooseRUO(boundaryThreshold_,
                                                 minOutlierPercentage_,
                                                 maxOutlierPercentage_);

  const bool REJECT_KNOWN_OUTLIERS = chooseRKO(minOutlierPercentage_,
                                               maxOutlierPercentage_);

  CV_Assert(!(maxOutlierPercentage < 0 || maxOutlierPercentage >= 1));
  CV_Assert(!(minOutlierPercentage < 0 || minOutlierPercentage >= 1));
  CV_Assert(minOutlierPercentage <= maxOutlierPercentage);

  const bool RETEST_OUTLIERS = retestOutliers;

  const int plateauCountThreshold = 1;

  //////////////////////////////////////////////////////////////////////////////
  // Step 1: map coordinates to joint image space.
  auto jointImageData = mapPerspective(img1, img2, NORMALIZE_DATA);

  //////////////////////////////////////////////////////////////////////////////
  // Step 2: apply perspective veronese map to data
  auto embedding = perspective_embedding(jointImageData, groupCount, false);
  auto veroneseData = embedding.getV().back();
  auto veroneseDerivative = embedding.getD().back();
  auto veroneseHessian = embedding.getH().back();

  //////////////////////////////////////////////////////////////////////////////
  // Step 3: Use influence function to perform robust pca

  Mat2D untrimmedData;
  Mat2D untrimmedVeronese;
  Mat3D untrimmedDerivative;
  Mat4D untrimmedHessian;
  size_t untrimmedSampleCount = 0;

  size_t sampleCount = static_cast<size_t>(jointImageData.cols);
  std::vector<InflVal> sortedIndex(sampleCount);
  std::vector<InflVal> influenceValues(sampleCount);

  Mat2D polynomialCoefficients;
  Mat2D polynomialNormals;

  if (REJECT_KNOWN_OUTLIERS || REJECT_UNKNOWN_OUTLIERS) {
    if (INFLUENCE_METHOD == InfluenceMethod::SAMPLE) {
      polynomialCoefficients = find_polynomials(veroneseData,
                                                     veroneseDerivative,
                                                     FITTING_METHOD,
                                                     1);

      // Reject outliers by the sample influence function
      for (int sIdx = 0; sIdx < static_cast<int>(sampleCount); sIdx++) {
        // compute the leave-one-out influence
        auto U = find_polynomials(veroneseData, veroneseDerivative,
                                  FITTING_METHOD, 1, sIdx);

        influenceValues[sIdx] =
            InflVal(subspace_angle(polynomialCoefficients, U), sIdx);
      }
    } else {
      // todo
    }

    sortedIndex = influenceValues;

    // Finally, reject outliers based on influenceValue
    std::sort(sortedIndex.begin(),
              sortedIndex.end(),
              sort_pred());

    untrimmedData = jointImageData.clone();
    untrimmedVeronese = veroneseData.clone();
    untrimmedDerivative = veroneseDerivative;
    untrimmedHessian = veroneseHessian;
    untrimmedSampleCount = sampleCount;
  }

  // REJECT_UNKNOWN_OUTLIERS
  EmbValT outlierStep = (sampleCount < 100) ?
                        round(1.0 / sampleCount * 100) / 100 : 0.01;

  size_t numDistanceStackElem = static_cast<size_t>(
      (maxOutlierPercentage - minOutlierPercentage)/outlierStep) + 1;

  std::vector<EmbValT> distanceStack(numDistanceStackElem);
  int plateauCount = 0;

  int outlierIndex = 0;
  std::vector<int> inlierIndices(sortedIndex.size());

  for (EmbValT outlierPercentage = minOutlierPercentage;
       outlierPercentage <= maxOutlierPercentage;
       outlierPercentage += outlierStep) {
    outlierIndex += 1;
    if (DEBUG > 0) {
      printf("Testing outlierPercentage %d",
             static_cast<int>(floor(100 * outlierPercentage)));
    }

    if (REJECT_UNKNOWN_OUTLIERS || REJECT_KNOWN_OUTLIERS) {
      sampleCount = static_cast<size_t>(floor(untrimmedSampleCount *
                                       (1 - outlierPercentage)));

      for (size_t i = 0; i < sampleCount; i++) {
        inlierIndices[i] = sortedIndex[sampleCount - 1 - i].second;
      }
      inlierIndices.resize(sampleCount);

      jointImageData = filterIdx2(untrimmedData, inlierIndices);
      veroneseData = filterIdx2(untrimmedVeronese, inlierIndices);
      veroneseDerivative = filterIdx3(untrimmedDerivative, inlierIndices);
      veroneseHessian = filterIdx4(untrimmedHessian, inlierIndices);
    }

    polynomialCoefficients = find_polynomials(veroneseData,
                                              veroneseDerivative,
                                              FITTING_METHOD, 1);

    ////////////////////////////////////////////////////////////////////////////
    // Step 4: Compute Derivatives and Hessians for the fitting polynomial.
    const int dimensionCount = jointImageData.rows;
    polynomialNormals = Mat2D::zeros(dimensionCount,
                                           static_cast<int>(sampleCount));

    for (int eIdx = 0; eIdx < dimensionCount; eIdx++) {
      polynomialNormals.row(eIdx) = polynomialCoefficients.t() *
          sliceIdx2(veroneseDerivative, eIdx);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Step 5: Estimate rejection percentage via Sampson distance

    if (REJECT_UNKNOWN_OUTLIERS) {
      distanceStack[outlierIndex] = 0;

      for (int inlierIndex = 0; inlierIndex < static_cast<int>(sampleCount);
           inlierIndex++) {
        Mat2D prod = polynomialCoefficients.t() * veroneseData.col(inlierIndex);
        EmbValT sampsonDistance = fabs(prod(0)) /
            norm(polynomialNormals.col(inlierIndex));

        if (sampsonDistance > distanceStack[outlierIndex]) {
          distanceStack[outlierIndex] = sampsonDistance;
        }

        if (sampsonDistance > boundaryThreshold && DEBUG <= 1) {
          break;
        }
      }

      if ((DEBUG <= 1) && (distanceStack[outlierIndex] <= boundaryThreshold)) {
        // One percentage below the boundary is found
        break;
      }
    }
  }

  const int dimensionCount = 5;

  Mat3D polynomialHessians = polyHessian(dimensionCount,
                                         static_cast<int>(sampleCount),
                                         polynomialCoefficients,
                                         veroneseHessian);

  //////////////////////////////////////////////////////////////////////////////
  // Step 6: Compute Tangent Spaces and Mutual Contractions. Use Mutual
  // Contractions as a similarity metric for spectral clustering. This
  // particular algorithm tries to oversegment the sample by using a
  // conservative tolerance.

  Mat2D t1, t2, t3, t4;
  t1 = polynomialCoefficients.t() * veroneseData;
  cv::pow(t1, 2., t2);
  cv::pow(polynomialNormals, 2., t3);
  reduce(t3, t4, 0, cv::REDUCE_SUM);

  std::vector<InflVal> normalLen(static_cast<size_t>(t2.cols));

  for (int i = 0; i < t2.cols; i++) {
    normalLen[i] = InflVal(t2(i)/t4(i), i);
  }

  std::sort(normalLen.begin(), normalLen.end(), sort_pred());

  std::vector<int> sortedIndices(static_cast<size_t>(t2.cols));
  for (int i = 0; i < t2.cols; i++) {
    sortedIndices[i] = normalLen[i].second;
  }

  jointImageData = filterIdx2(jointImageData, sortedIndices);
  polynomialNormals = filterIdx2(polynomialNormals, sortedIndices);
  polynomialHessians = filterIdx3(polynomialHessians, sortedIndices);

  auto emb = perspective_embedding(jointImageData, 1, false);
  auto embeddedData = emb.getV().back();
  auto DembeddedData = emb.getD().back();
  auto HembeddedData = emb.getH().back();

  IndexMat2D labels = IndexMat2D::ones(1, static_cast<int>(sampleCount))*-1;

  // todo: find out required size
  IndexMat2D clusterPrototype = Mat2D::zeros(1, static_cast<int>(sampleCount));
  int clusterCount = 1;

  labels(0) = 0;
  clusterPrototype(0) = 0;
  EmbValT cosTolerance = cos(angleTolerance);

  for (int sampleIdx = 1; sampleIdx < static_cast<int>(sampleCount);
       sampleIdx++) {
    for (int clusterIdx = 0; clusterIdx < clusterCount; clusterIdx++) {
      std::vector<int> indices {sampleIdx, clusterPrototype(clusterIdx)};

      Mat2D U;
      jacobiSVD_U(filterIdx2(polynomialNormals, indices), &U);

      Mat2D T = U.colRange(2, U.cols);

      Mat2D C1 = T.t() * sliceIdx3(polynomialHessians,
                                   sampleIdx) * T;

      Mat2D C2 = T.t() * sliceIdx3(polynomialHessians,
                                   clusterPrototype(clusterIdx)) * T;

      auto C1_ = C1.reshape(0, 1);
      auto C2_ = C2.reshape(0, C2.rows*C2.cols);

      Mat2D Cp = cv::abs(C1_/cv::norm(C1_) * C2_/cv::norm(C2_));

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

  //////////////////////////////////////////////////////////////////////////////
  // Step 7: Recover quadratic matrices for each cluster.

  if (clusterCount < static_cast<int>(groupCount)) {
    std::runtime_error("Too few motion models found. "\
                       "Please choose a smaller angleTolerance");
  }

  Mat3D quadratics = Mat3D_zeros(dimensionCount, dimensionCount, clusterCount);

  for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
    std::vector<int> currIndices;
    for (int i = 0; i < labels.cols; i++) {
      if (labels(i) == clusterIndex) {
        currIndices.push_back(i);
      }
    }

    int clusterSize = static_cast<int>(currIndices.size());
    auto clusterData = filterIdx2(embeddedData, currIndices);
    auto clusterHessian = filterIdx4(HembeddedData, currIndices);

    Mat2D U;
    jacobiSVD_U(clusterData, &U);

    Mat3D hpn = polyHessian(dimensionCount,
                            clusterSize,
                            U.col(U.cols-1),
                            clusterHessian);

    auto quad = meanIdx3(hpn);
    if (NORMALIZE_QUADRATICS) {
      quad /= cv::norm(quad);
    }

    for (int j = 0; j < dimensionCount; j++) {
      for (int i = 0; i < dimensionCount; i++) {
        quadratics[j](i, clusterIndex) = quad(j, i);
      }
    }
  }

  std::cout << quadratics << std::endl;

  return veroneseData;
}
