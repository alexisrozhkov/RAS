#include <vector>
#include "balls_and_bins.h"


typedef cv::Mat1i Cell;
typedef std::vector<Cell> CellArray;
typedef std::vector<CellArray> CellArray2D;


CellArray balls_and_bins(const unsigned int ballCount,
                         const unsigned int binCount,
                         const bool all) {
  // Create a two dimensional cell array to hold solutions for smaller values of both ballCount and binCount.
  CellArray2D outputCellArray2D;
  for(int j = 0; j < ballCount; j++) {
    outputCellArray2D.push_back(CellArray(binCount));
  }

  // The top row of the cell array is given by identity matrices.
  for(int column = 0; column < binCount; column++) {  // Column of the cell array, that is.
    outputCellArray2D[0][column] = Cell::eye(column+1, column+1);
  }

  // The leftmost column of cell array (1 Bin) is also trivial.
  for(int row = 0; row < ballCount; row++) {
    outputCellArray2D[row][0] = Cell::ones(1, 1)*(row+1);
  }

  // Apply method of computing that exploits a pattern that becomes noticeable when the
  // arrangements are written out in increasing order.  The base cases for one ball and for
  // one bin are trivial, and the rest of the cases can be formed by induction in two directions.
  for(int row = 1; row < ballCount; row++) {
    for(int column = 1; column < binCount; column++) {
      // Increment the first column of the case for one less ball.
      Cell upper = outputCellArray2D[row - 1][column].clone();
      upper.colRange(0, 1) += Cell::ones(upper.rows, 1);

      // Prepend a column of zeros to the case for one less bin.
      Cell lower;
      const Cell& temp2 = outputCellArray2D[row][column - 1];
      cv::hconcat(Cell::zeros(temp2.rows, 1), temp2, lower);

      // The case for one more bin is just the vertical concatenation.
      cv::vconcat(upper, lower, outputCellArray2D[row][column]);
    }
  }

  CellArray outputCellArray;

  if(!all) {
    outputCellArray.push_back(outputCellArray2D.back().back());
  }

  else {
    for(int i = 0; i < ballCount; i++) outputCellArray.push_back(outputCellArray2D[i].back());
  }

  return outputCellArray;
}
