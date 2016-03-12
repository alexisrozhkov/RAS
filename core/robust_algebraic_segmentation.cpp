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

static const int dimensionCount = 5;

EmbValT chooseMiOP(const EmbValT miOP, const EmbValT maOP) {
  const EmbValT defaultVal = 0;

  if (miOP == NotSpecified) {  // nothing given
    return defaultVal;
  } else {  // one or two values given
    return miOP;
  }
}

EmbValT chooseMaOP(const EmbValT miOP, const EmbValT maOP) {
  const EmbValT defaultVal = 0.5;

  if (miOP == NotSpecified) {  // nothing given
    return defaultVal;
  } else if (maOP == NotSpecified) {  // one value given
    return miOP;
  } else {
    return maOP;  // range given
  }
}

bool chooseRKO(const EmbValT miOP, const EmbValT maOP) {
  return (miOP > 0) && (maOP == NotSpecified);
}

bool chooseRUO(const EmbValT boundaryThreshold,
               const EmbValT miOP, const EmbValT maOP) {
  return boundaryThreshold != NotSpecified ||
      ((miOP != NotSpecified) && (maOP != NotSpecified));
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

void clusterQuad(const IndexMat2D &labels,
                 const int clusterIdx,
                 const int dimensionCount,
                 const Mat2D &embeddedData,
                 const Mat4D &HembeddedData,
                 const bool NORMALIZE_QUADRATICS,
                 Mat3D *quadratics) {
  std::vector<int> currIndices;
  for (int i = 0; i < labels.cols; i++) {
    if (labels(i) == clusterIdx) {
      currIndices.push_back(i);
    }
  }

  int clusterSize = static_cast<int>(currIndices.size());
  auto clusterData = filterIdx2(embeddedData, currIndices);
  auto clusterHessian = filterIdx4(HembeddedData, currIndices);

  Mat2D U;
  armaSVD_U(clusterData, &U);

  Mat3D hpn = polyHessian(dimensionCount,
                          clusterSize,
                          U.col(U.cols - 1),
                          clusterHessian);

  Mat2D quad = meanIdx3(hpn);
  if (NORMALIZE_QUADRATICS) {
    quad /= cv::norm(quad);
  }

  for (int j = 0; j < dimensionCount; j++) {
    for (int i = 0; i < dimensionCount; i++) {
      (*quadratics)[j](i, clusterIdx) = quad(j, i);
    }
  }
}

void step8(const unsigned int groupCount,
           const bool NORMALIZE_QUADRATICS,
           const Mat2D &jointImageData,
           const cv::Mat_<EmbValT> &embeddedData,
           const std::vector<Mat3D> &HembeddedData,
           IndexMat2D &labels,
           int clusterCount,
           Mat3D &quadratics) {
  while (clusterCount > static_cast<int>(groupCount)) {
    std::vector<int> cSampleCounts(static_cast<size_t>(clusterCount));
    for (int i = 0; i < labels.cols; i++) {
      cSampleCounts[labels(i)] += 1;
    }

    int minVal = std::numeric_limits<int>::max(), smallestClust = 0;
    for (int i = 0; i < clusterCount; i++) {
      if (cSampleCounts[i] < minVal) {
        minVal = cSampleCounts[i];
        smallestClust = i;
      }
    }

    IndexMat2D clusterChanged = IndexMat2D::zeros(1, clusterCount-1);

    if (smallestClust != clusterCount-1) {
      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == smallestClust) {
          labels(i) = -1;
        }
      }

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == clusterCount-1) {
          labels(i) = smallestClust;
        }
      }

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == -1) {
          labels(i) = clusterCount-1;
        }
      }

      auto temp = sliceIdx3(quadratics, smallestClust);

      auto quadraticsCopy = quadratics;
      for (int j = 0; j < dimensionCount; j++) {
        for (int i = 0; i < dimensionCount; i++) {
          quadratics[j](i, smallestClust) = quadraticsCopy[j](i, clusterCount-1);
        }
      }

      for (int j = 0; j < dimensionCount; j++) {
        for (int i = 0; i < dimensionCount; i++) {
          quadratics[j](i, clusterCount-1) = temp(j, i);
        }
      }
    }

    {
      std::vector<int> sampleIdx;
      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == clusterCount - 1) {
          sampleIdx.push_back(i);
        }
      }

      for(size_t j = 0; j < sampleIdx.size(); j++) {
        int minIdx = 0;
        EmbValT minDist = std::numeric_limits<EmbValT>::max();
        Mat2D distance = minDist * Mat2D::ones(1, clusterCount - 1);

        for (int clusterIndex = 0; clusterIndex < clusterCount - 1;
             clusterIndex++) {
          Mat2D u1 = jointImageData.col(sampleIdx[j]);//sliceIdx2(jointImageData, sampleIdx[j]);
          Mat2D u2 = sliceIdx3(quadratics, clusterIndex);
          Mat2D u3 = u1.t() * u2 * u1;

          distance(clusterIndex) = fabs(u3(0));

          if (distance(clusterIndex) < minDist) {
            minDist = distance(clusterIndex);
            minIdx = clusterIndex;
          }
        }

        //for (int i = 0; i < sampleIdx.size(); i++) {
          labels(sampleIdx[j]) = minIdx;
          clusterChanged(labels(sampleIdx[j])) = 1;
        //}
      }
    }

    clusterCount -= 1;

    for (int clusterIdx = 0; clusterIdx < clusterCount;
         clusterIdx++) {
      if (!clusterChanged(clusterIdx)) {
        continue;
      }

      clusterQuad(labels,
                  clusterIdx,
                  dimensionCount,
                  embeddedData,
                  HembeddedData,
                  NORMALIZE_QUADRATICS,
                  &quadratics);
    }
  }
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

Mat2D robust_algebraic_segmentation(const Mat2D &img1Unnorm,
                                    const Mat2D &img2Unnorm,
                                    const unsigned int groupCount,

                                    const EmbValT bT,
                                    const EmbValT miOP,
                                    const EmbValT maOP,

                                    const int debug,
                                    const bool postRansac,
                                    const std::vector<EmbValT> angleTolerance,
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

  for (size_t i = 0; i < angleTolerance.size(); i++) {
    CV_Assert(!(angleTolerance[i] < 0 || angleTolerance[i] >= CV_PI / 2));
  }

  // todo: handle unspecified value of boundaryThreshold more gracefully
  const EmbValT boundaryThreshold = (bT == NotSpecified) ? 0 : bT;

  // will not complain if boundaryThreshold_ was -1, since it's a default value
  // and will be overwritten with 0
  CV_Assert(boundaryThreshold >= 0);

  const FindPolyMethod FITTING_METHOD = fittingMethod;
  const InfluenceMethod INFLUENCE_METHOD = influenceMethod;

  // todo: check if it is an error/typo in Matlab implementation
  // const bool NORMALIZE_COORDINATES = true;
  const bool NORMALIZE_DATA = normalizeCoordinates;

  const bool NORMALIZE_QUADRATICS = normalizeQuadratics;

  const EmbValT minOutlierPercentage = chooseMiOP(miOP, maOP);
  const EmbValT maxOutlierPercentage = chooseMaOP(miOP, maOP);

  const bool REJECT_UNKNOWN_OUTLIERS = chooseRUO(bT, miOP, maOP);
  const bool REJECT_KNOWN_OUTLIERS = chooseRKO(miOP, maOP);

  CV_Assert(!(maxOutlierPercentage < 0 || maxOutlierPercentage >= 1));
  CV_Assert(!(minOutlierPercentage < 0 || minOutlierPercentage >= 1));
  CV_Assert(minOutlierPercentage <= maxOutlierPercentage);

  const bool RETEST_OUTLIERS = retestOutliers;

  const int plateauCountThreshold = 1;

  //////////////////////////////////////////////////////////////////////////////
  // Step 1: map coordinates to joint image space.
  int sampleCount = img1Unnorm.cols;

  const Mat2D img1 = normalizeCoords(img1Unnorm, NORMALIZE_DATA);
  const Mat2D img2 = normalizeCoords(img2Unnorm, NORMALIZE_DATA);

  Mat2D jointImageData = getJointData(img1, img2);

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
  int untrimmedSampleCount = 0;

  std::vector<InflVal> sortedIndex(static_cast<size_t>(sampleCount));
  std::vector<InflVal> influenceValues(static_cast<size_t>(sampleCount));

  Mat2D polynomialCoefficients;
  Mat2D polynomialNormals;

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

  EmbValT outlierPercentage;

  for (outlierPercentage = minOutlierPercentage;
       outlierPercentage <= maxOutlierPercentage;
       outlierPercentage += outlierStep) {
    outlierIndex += 1;
    if (DEBUG > 0) {
      printf("Testing outlierPercentage %d",
             static_cast<int>(floor(100 * outlierPercentage)));
    }

    if (REJECT_UNKNOWN_OUTLIERS || REJECT_KNOWN_OUTLIERS) {
      sampleCount = static_cast<int>(floor(untrimmedSampleCount *
                                     (1 - outlierPercentage)));

      for (int i = 0; i < sampleCount; i++) {
        inlierIndices[i] = sortedIndex[sampleCount - 1 - i].second;
      }
      inlierIndices.resize(static_cast<size_t>(sampleCount));

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
    polynomialNormals = Mat2D::zeros(dimensionCount, sampleCount);

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

  Mat3D polynomialHessians = polyHessian(dimensionCount,
                                         sampleCount,
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
  cv::reduce(t3, t4, 0, cv::REDUCE_SUM);

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

  const int angleCount = static_cast<int>(angleTolerance.size());
  IndexMat2D allLabels = IndexMat2D::ones(angleCount,
                                          static_cast<int>(sampleCount))*-1;

  IndexMat2D labels;

  for (int angleIndex = 0; angleIndex < angleCount; angleIndex++) {
    labels = IndexMat2D::ones(1, static_cast<int>(sampleCount))*-1;

    int clusterCount = 0;
    labels(0) = 0;

    IndexMat2D clusterPrototype = Mat2D::zeros(1, sampleCount);
    clusterPrototype(0) = 0;

    EmbValT cosTolerance = cos(angleTolerance[angleIndex]);

    for (int sampleIdx = 1; sampleIdx < sampleCount; sampleIdx++) {
      for (int clusterIdx = 0; clusterIdx < clusterCount; clusterIdx++) {
        std::vector<int> indices{sampleIdx, clusterPrototype(clusterIdx)};

        Mat2D U;
        armaSVD_U(filterIdx2(polynomialNormals, indices), &U);

        Mat2D T = U.colRange(2, U.cols);

        Mat2D C1 = T.t() * sliceIdx3(polynomialHessians,
                                     sampleIdx) * T;

        Mat2D C2 = T.t() * sliceIdx3(polynomialHessians,
                                     clusterPrototype(clusterIdx)) * T;

        auto C1_ = C1.reshape(0, 1);
        auto C2_ = C2.reshape(0, C2.rows * C2.cols);

        Mat2D Cp = cv::abs(C1_ / cv::norm(C1_) * C2_ / cv::norm(C2_));

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

    Mat3D
        quadratics = Mat3D_zeros(dimensionCount, dimensionCount, clusterCount);

    for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
      clusterQuad(labels,
                  clusterIndex,
                  dimensionCount,
                  embeddedData,
                  HembeddedData,
                  NORMALIZE_QUADRATICS,
                  &quadratics);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Step 8: Kill off smaller clusters one at a time, reassigning samples
    // to remaining clusters with minimal algebraic distance.

    step8(groupCount,
          NORMALIZE_QUADRATICS,
          jointImageData,
          embeddedData,
          HembeddedData,
          labels,
          clusterCount,
          quadratics);

    auto labelsCopy = labels.clone();
    for (int i = 0; i < sampleCount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ; i++) {
      labels(sortedIndices[i]) = labelsCopy(i);
    }

    labels.copyTo(allLabels.row(angleIndex));
  }

  std::cout << labels+1 << std::endl;
  //return labels;

  if (angleCount != 1) {
    // todo:
    //std::cout << labels+1 << std::endl;
    // labels = aggregate_labels(allLabels, groupCount);
    //std::cout << labels+1 << std::endl;
  }

  if (REJECT_UNKNOWN_OUTLIERS || REJECT_KNOWN_OUTLIERS) {
    IndexMat2D untrimmedAllLabels = -1*IndexMat2D::ones(angleCount,
                                        untrimmedSampleCount);

    for (int i = 0; i < static_cast<int>(inlierIndices.size()); i++) {
      allLabels.col(i).copyTo(untrimmedAllLabels.col(inlierIndices[i]));
    }

    allLabels = untrimmedAllLabels;

    IndexMat2D untrimmedLabels = -1*IndexMat2D::ones(1, untrimmedSampleCount);

    for (int i = 0; i < static_cast<int>(inlierIndices.size()); i++) {
      untrimmedLabels(inlierIndices[i]) = labels(i);
    }

    labels = untrimmedLabels;

    jointImageData = untrimmedData;
    sampleCount = untrimmedSampleCount;
  }

  if (REJECT_UNKNOWN_OUTLIERS && outlierPercentage >= maxOutlierPercentage) {
    std::cout << "The RHQA function did not find an mixed motion model "\
                 "within the boundary threshold." << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Step 9: Post-refinement using RANSAC
  // For each group, run RANSAC to re-estimate a motion model.
  // After the refinement, resegment the data based on the motion model.

  if (POST_RANSAC) {
    const int RANSACIteration = 10;
    const EmbValT FThreshold = 0.035;  // FThreshold = 0.015;
    const EmbValT HAngleThreshold = 0.04;
    const int FSampleSize = 8;
    const int HSampleSize = 4;
    IndexMat2D currentLabels = -1*IndexMat2D::ones(1, sampleCount);

    auto normedVector2 = img2.clone();
    for (int sampleIdx = 0; sampleIdx < sampleCount; sampleIdx++) {
      normedVector2.col(sampleIdx) /= cv::norm(normedVector2.col(sampleIdx));
    }

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

    //std::cout << kronImg1Img2 << std::endl;

    IndexMat2D isGroupOutliers = IndexMat2D::zeros(groupCount, sampleCount);
    Mat3D bestF = Mat3D_zeros(3, 3, groupCount);
    Mat3D bestH = Mat3D_zeros(3, 3, groupCount);

    std::vector<int> maxFGroupDataIndices;

    for (int grpIdx = 0; grpIdx < static_cast<int>(groupCount); grpIdx++) {
      std::vector<int> groupDataIndices;

      for (int i = 0; i < labels.cols; i++) {
        if (labels(i) == grpIdx) {
          groupDataIndices.push_back(i);
        }
      }

      int maxFConsensus = 0;
      IndexMat2D maxFConsensusIndices;

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
        // Sample 8 points
        std::random_shuffle(groupDataIndices.begin(), groupDataIndices.end());

        std::vector<int> hypIndices(groupDataIndices.begin(),
                                    groupDataIndices.begin()+FSampleSize);

        // RANSAC an epipolar model
        Mat2D A = filterIdx2(kronImg1Img2, hypIndices).t();

        Mat2D V;
        armaSVD_V(A, &V);

        Mat2D Fstacked = V.col(V.cols-1).clone();
        Mat2D F = Fstacked.reshape(0, 3).t();

        // Step 2: Compute a consensus among all group samples
        IndexMat2D currentConsensusIndices = IndexMat2D::zeros(1, sampleCount);
        Mat2D normedVector1 = F * filterIdx2(img1, groupDataIndices);

        Mat2D normedVector1sq, sums;
        cv::pow(normedVector1, 2, normedVector1sq);
        cv::reduce(normedVector1sq, sums, 0, cv::REDUCE_SUM);
        cv::pow(sums, 0.5, sums);

        for(int i = 0; i < normedVector1.cols; i++) {
          normedVector1.col(i) /= sums(i);
        }

        Mat2D temp;
        cv::multiply(filterIdx2(normedVector2, groupDataIndices),
                     normedVector1,
                     temp);

        Mat2D consensus;
        cv::reduce(temp, consensus, 0, cv::REDUCE_SUM);

        //todo: replace with indval
        temp = (cv::abs(consensus) < FThreshold)/255;

        int currentConsensus = 0;
        int idx = 0;
        for(int i = 0; i < groupDataIndices.size(); i++) {
          currentConsensusIndices(groupDataIndices[i]) = (int)temp(i);
          currentConsensus += (int)temp(i);
        }

        if (currentConsensus > maxFConsensus) {
          maxFConsensus = currentConsensus;
          maxFConsensusIndices = currentConsensusIndices;

          maxFGroupDataIndices = std::vector<int>(groupDataIndices.begin(),
                                                  groupDataIndices.begin()+FSampleSize);

          for(int j = 0; j < 3; j++) {
            for(int i = 0; i < 3; i++) {
              bestF[j](i, grpIdx) = F(j, i);
            }
          }

/*
          std::cout << maxFConsensus << std::endl;
          std::cout << maxFConsensusIndices << std::endl;

          for(int i = 0; i < maxFGroupDataIndices.size(); i++) {
            std::cout << maxFGroupDataIndices[i]+1 << " ";
          }
          std::cout << std::endl;

          std::cout << sliceIdx3(bestF, grpIdx) << std::endl;
*/
        }
      }

      std::cout << maxFConsensus << std::endl;

      return Mat2D();
    }
  }

  return veroneseData;
}
