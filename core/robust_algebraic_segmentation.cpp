// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <core/perspective_embedding.h>
#include <core/utils/cholesky.h>
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

Mat2D filterIdx2(const Mat2D &src, const std::vector<int> &indices) {
  Mat2D out(src.rows, static_cast<int>(indices.size()));

  for (int i = 0; i < static_cast<int>(indices.size()); i++) {
    src.col(indices[i]).copyTo(out.col(i));
  }

  return out;
}

Mat3D filterIdx3(const Mat3D &src, const std::vector<int> &indices) {
  Mat3D out = src;

  for (size_t j = 0; j < src.size(); j++) {
    out[j] = Mat2D(src[j].rows, static_cast<int>(indices.size()));

    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      src[j].col(indices[i]).copyTo(out[j].col(i));
    }
  }

  return out;
}

Mat4D filterIdx4(const Mat4D &src, const std::vector<int> &indices) {
  Mat4D out = src;

  for (size_t k = 0; k < src.size(); k++) {
    for (size_t j = 0; j < src[0].size(); j++) {
      out[k][j] = Mat2D(src[k][j].rows,
                        static_cast<int>(indices.size()));

      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        src[k][j].col(indices[i]).copyTo(out[k][j].col(i));
      }
    }
  }

  return out;
}

Mat2D sliceIdx2(const Mat3D &src, const int idx) {
  Mat2D temp(static_cast<int>(src.size()), src[0].cols);

  for (int j = 0; j < static_cast<int>(src.size()); j++) {
    for (int i = 0; i < src[0].cols; i++) {
      temp(j, i) = src[j](idx, i);
    }
  }

  return temp;
}

Mat2D sliceIdx23(const Mat4D &src, const int idx1, const int idx2) {
  Mat2D temp(static_cast<int>(src.size()), src[0][0].cols);

  for (int j = 0; j < static_cast<int>(src.size()); j++) {
    for (int i = 0; i < src[0][0].cols; i++) {
      temp(j, i) = src[j][idx1](idx2, i);
    }
  }

  return temp;
}

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
  const bool defaultVal = false;

  if ((minOutlierPercentage > 0) &&
      (maxOutlierPercentage == NotSpecified)) {
      return true;  // one value given and its positive
  }

  return defaultVal;
}

bool chooseRUO(const EmbValT boundaryThreshold,
               const EmbValT minOutlierPercentage,
               const EmbValT maxOutlierPercentage) {
  const bool defaultVal = false;

  if (boundaryThreshold != NotSpecified) {
    return true;  // boundary threshold specified
  }

  if ((minOutlierPercentage != NotSpecified) &&
      (maxOutlierPercentage != NotSpecified)) {
    return true;  // outlier range specified
  }

  return defaultVal;
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
  CV_Assert(!(boundaryThreshold < 0));

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
  CV_Assert(!(minOutlierPercentage > maxOutlierPercentage));

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
  size_t untrimmedSampleCount;

  size_t sampleCount = static_cast<size_t>(jointImageData.cols);
  std::vector<InflVal> sortedIndex(sampleCount);
  std::vector<InflVal> influenceValues(sampleCount);

  Mat2D polynomialCoefficients;

  if (REJECT_KNOWN_OUTLIERS || REJECT_UNKNOWN_OUTLIERS) {
    if (INFLUENCE_METHOD == InfluenceMethod::SAMPLE) {
      polynomialCoefficients = find_polynomials(veroneseData,
                                                     veroneseDerivative,
                                                     FITTING_METHOD,
                                                     1);

      // Reject outliers by the sample influence function
      for (int sIdx = 0; sIdx < sampleCount; sIdx++) {
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

      for (int i = 0; i < sampleCount; i++) {
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
    Mat2D polynomialNormals = Mat2D::zeros(dimensionCount,
                                           static_cast<int>(sampleCount));

    for (int eIdx = 0; eIdx < dimensionCount; eIdx++) {
      polynomialNormals.row(eIdx) = polynomialCoefficients.t() *
          sliceIdx2(veroneseDerivative, eIdx);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Step 5: Estimate rejection percentage via Sampson distance

    if (REJECT_UNKNOWN_OUTLIERS) {
      distanceStack[outlierIndex] = 0;

      for (int inlierIndex = 0; inlierIndex < sampleCount; inlierIndex++) {
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

  // todo: use Mat3D_zeros here
  Mat3D polynomialHessians(dimensionCount);
  for (int eIdx1 = 0; eIdx1 < dimensionCount; eIdx1++) {
    polynomialHessians[eIdx1] = Mat2D(dimensionCount,
                                      static_cast<int>(sampleCount));
  }

  for (int eIdx1 = 0; eIdx1 < dimensionCount; eIdx1++) {
    for (int eIdx2 = eIdx1; eIdx2 < dimensionCount; eIdx2++) {
      polynomialHessians[eIdx1].row(eIdx2) = polynomialCoefficients.t() *
          sliceIdx23(veroneseHessian, eIdx1, eIdx2);

      if (eIdx1 != eIdx2) {
        polynomialHessians[eIdx1].row(eIdx2).copyTo(
            polynomialHessians[eIdx2].row(eIdx1));
      }
    }
  }

  std::cout << polynomialHessians << std::endl;

  return veroneseData;
}
