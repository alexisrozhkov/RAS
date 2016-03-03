// Copyright 2016 Alexey Rozhkov

#include <core/utils/balls_and_bins.h>


IndexMat2DArray balls_and_bins(const unsigned int ballCount,
                               const unsigned int binCount,
                               const bool all) {
  // Create a two dimensional cell array to hold solutions for smaller values
  // of both ballCount and binCount.
  IndexMat2DArray2D outputCellArray2D;
  for (uint j = 0; j < ballCount; j++) {
    outputCellArray2D.push_back(IndexMat2DArray(binCount));
  }

  // The top row of the cell array is given by identity matrices.
  for (uint column = 0; column < binCount; column++) {
    outputCellArray2D[0][column] = IndexMat2D::eye(column + 1, column + 1);
  }

  // The leftmost column of cell array (1 Bin) is also trivial.
  for (uint row = 0; row < ballCount; row++) {
    outputCellArray2D[row][0] = IndexMat2D::ones(1, 1) * (row + 1);
  }

  // Apply method of computing that exploits a pattern that becomes noticeable
  // when the arrangements are written out in increasing order.
  // The base cases for one ball and for one bin are trivial, and the rest
  // of the cases can be formed by induction in two directions.
  for (uint row = 1; row < ballCount; row++) {
    for (uint column = 1; column < binCount; column++) {
      // Increment the first column of the case for one less ball.
      IndexMat2D upper = outputCellArray2D[row - 1][column].clone();
      upper.colRange(0, 1) += IndexMat2D::ones(upper.rows, 1);

      // Prepend a column of zeros to the case for one less bin.
      IndexMat2D lower;
      const IndexMat2D &temp2 = outputCellArray2D[row][column - 1];
      cv::hconcat(IndexMat2D::zeros(temp2.rows, 1), temp2, lower);

      // The case for one more bin is just the vertical concatenation.
      cv::vconcat(upper, lower, outputCellArray2D[row][column]);
    }
  }

  IndexMat2DArray outputCellArray;

  if (!all) {
    outputCellArray.push_back(outputCellArray2D.back().back());
  } else {
    for (uint i = 0; i < ballCount; i++) {
      outputCellArray.push_back(outputCellArray2D[i].back());
    }
  }

  return outputCellArray;
}
