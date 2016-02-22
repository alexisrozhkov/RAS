#include <vector>
#include "balls_and_bins.h"


typedef cv::Mat1i Cell;
typedef std::vector<Cell> CellArray;
typedef std::vector<CellArray> CellArray2D;


CellArray balls_and_bins(const unsigned int ballCount,
                         const unsigned int binCount,
                         bool all) {
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

  // Apply method of computing that exploits a pattern that becomes noticable when the
  // arrangements are written out in increasing order.  The base cases for one ball and for
  // one bin are trivial, and the rest of the cases can be formed by induction in two directions.
  for(int row = 1; row < ballCount; row++) {
    for(int column = 1; column < binCount; column++) {
      const Cell& temp1 = outputCellArray2D[row - 1][column];
      const Cell& temp2 = outputCellArray2D[row][column - 1];
      Cell& curr = outputCellArray2D[row][column];

      curr = Cell::zeros(temp1.rows + temp2.rows, temp1.cols);

      for (int j = 0; j < temp1.rows; j++) {
        curr(j, 0) = temp1(j, 0) + 1;
      }

      for (int i = 1; i < temp1.cols; i++) {
        for (int j = 0; j < temp1.rows; j++) {
          curr(j, i) = temp1(j, i);
        }

        for (int j = 0; j < temp2.rows; j++) {
          curr(temp1.rows + j, i) = temp2(j, i - 1);
        }
      }
    }
  }

  CellArray outputCellArray;

  if(!all) {
    // (converting) return last cell
    outputCellArray.push_back(outputCellArray2D[ballCount-1][binCount-1]);
  }

  else {
    // (converting) return last column
    for(int i = 0; i < ballCount; i++) outputCellArray.push_back(outputCellArray2D[i][binCount-1]);
  }

  return outputCellArray;
}
