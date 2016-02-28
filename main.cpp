#include <iostream>
#include <perspective_embedding.h>

using namespace std;

// Matlab-friendly notation for testing
// matchesData = [1; 2; 3; 4; 5]
// matchesData = [1, 6; 2, 7; 3, 8; 4, 9; 5, 10]
// matchesData = [0; 2; 3; 4; 5]
// matchesData = [0, 6; 2, 0; 3, 8; 4, 9; 5, 10]

int main() {
  // double matchesData51[] = {1, 2, 3, 4, 5};
  double matchesData52[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};
  // double matchesData051[] = {0, 2, 3, 4, 5};
  // double matchesData052[] = {0, 6, 2, 0, 3, 8, 4, 9, 5, 10};

  std::cout << perspective_embedding(Mat2D(5, 2, matchesData52), 1) << std::endl;

  return 0;
}

