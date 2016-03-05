// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <iostream>

// Matlab-friendly notation for testing

/*

*/

int main() {
  double img1[] = {
    0.326301, 0.516315, 0.765559, 0.980400, 0.327472, 0.344081, 0.540702, 0.702417, 0.795220, 0.557404,
    0.500971, 0.367573, 0.544737, 0.088109, 0.130662, 0.286203, 0.352365, 0.215530, 0.298761, 0.966660
  };

  double img2[] = {
    0.73398, 0.33626, 0.88408, 0.47326, 0.74193, 0.12163, 0.56227, 0.29542, 0.74600, 0.92105,
    0.94321, 0.53112, 0.93380, 0.28435, 0.44381, 0.51287, 0.45738, 0.55943, 0.74321, 0.17671
  };

  auto mat1 = Mat2D(2, 10, img1);
  auto mat2 = Mat2D(2, 10, img2);

  auto p = robust_algebraic_segmentation(mat1, mat2, 1);

  // std::cout << p << std::endl;

  return 0;
}

