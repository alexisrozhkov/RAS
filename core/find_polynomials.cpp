// Copyright 2016 Alexey Rozhkov

#include <core/utils/arma_wrapper.h>
#include <core/find_polynomials.h>
#include <vector>


// return the input without column with index = idx
Mat2D throwOneOut(const Mat2D &input, const int idx) {
  if (idx == -1) {
    return input.clone();
  }

  Mat2D out(input.rows, input.cols-1);

  // handle border cases correctly
  if (idx == 0) {
    input.colRange(1, input.cols).copyTo(out.colRange(0, input.cols-1));

    return out;
  } else if (idx == input.cols-1) {
    input.colRange(0, input.cols-1).copyTo(out.colRange(0, input.cols-1));

    return out;
  } else {
    input.colRange(0, idx).copyTo(out.colRange(0, idx));
    input.colRange(idx + 1, input.cols).copyTo(
        out.colRange(idx, input.cols - 1));

    return out;
  }
}

Mat2D find_polynomials(const EmbeddingData &embedding,
                       const FindPolyMethod method,
                       const int ignoreSample) {
  if (method == FindPolyMethod::FISHER) {
    const EmbValT RAYLEIGHQUOTIENT_EPSILON = 10;

    // if ignoreSample == -1 returns data_ unmodified
    Mat2D data = throwOneOut(embedding.getV(), ignoreSample);
    const Mat3D &derivative = embedding.getD();

    const int veroneseDimension = (uint) derivative.size(),
        dimensionCount = derivative[0].rows,
        sampleCount = data.cols;

    Mat2D B = Mat2D::zeros(veroneseDimension, veroneseDimension);
    Mat2D A = data * data.t();

    for (int dimensionIndex = 0; dimensionIndex < dimensionCount;
         dimensionIndex++) {
      Mat2D temp(veroneseDimension, sampleCount);
      for (int j = 0; j < veroneseDimension; j++) {
        int idx = 0;
        for (int i = 0; i < derivative[0].cols; i++) {
          temp(j, idx) = derivative[j](dimensionIndex, i);

          // if we come across column with idx == ignoreSample, then
          // overwrite it in next iteration
          if (i != ignoreSample) idx++;
        }
      }

      B += temp * temp.t();
    }

    // according to the paper denominator should be regularized.
    // but in the reference implementation nominator is.
    // todo: check this more closely or maybe contact the author
    A += RAYLEIGHQUOTIENT_EPSILON * Mat2D::eye(veroneseDimension,
                                               veroneseDimension);

    Mat2D alphas, betas;
    generalizedSchur(A, B, &alphas, &betas);

    // find eigenvalues from ratios and sort
    // perhaps special care has to be taken for cases when beta ~ 0
    std::vector<EmbValT> eigenvals((uint) alphas.rows);

    for (int i = 0; i < alphas.rows; i++) {
      eigenvals[i] = fabs(alphas(i)) / fabs(betas(i));
    }

    std::sort(eigenvals.begin(),  // NOLINT(build/include_what_you_use)
              eigenvals.end());

    Mat2D out(veroneseDimension, 1);
    cv::SVD::solveZ(A - eigenvals[0] * B, out);

    return out;
  } else {
    CV_Assert(ignoreSample == -1);  // ignoreSample is not currently handled
    Mat2D w, u, vt;
    cv::SVD::compute(embedding.getV(), w, u, vt, cv::SVD::FULL_UV);
    return u.col(u.cols - 1);
  }
}
