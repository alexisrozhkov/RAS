// Copyright 2016 Alexey Rozhkov

#include <core/robust_algebraic_segmentation.h>
#include <tests/core_tests/robust_algebraic_segmentation_data.h>

#include <iostream>


int main() {
  auto segm = robust_algebraic_segmentation(Mat2D(img1, true).reshape(0, 2),
                                            Mat2D(img2, true).reshape(0, 2),
                                            2, RAS_params(0.25, {3*CV_PI/180}));

  std::cout << segm.labels+1 << std::endl;

  return 0;
}

