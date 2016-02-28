#include <iostream>
#include <perspective_embedding.h>
#include <find_polynomials.h>

using namespace std;

// Matlab-friendly notation for testing
// matchesData = [100, 200, 300, 400, 500, 600, 700, 800, 900; 150, 250, 350, 450, 550, 650, 750, 850, 950; 200, 300, 400, 500, 600, 700, 800, 900,1000; 350, 450, 550, 650, 750, 850, 950,1050,1150; 1,   1,   1,   1,   1,   1,   1,   1,   1]


int main() {
  // this should be non-degenerate - planar translation with dx = 100, dy = 200
  double matchesData[] = {100, 200, 300, 400, 500, 600, 700, 800, 900,
                          150, 250, 350, 450, 550, 650, 750, 850, 950,
                          200, 300, 400, 500, 600, 700, 800, 900,1000,
                          350, 450, 550, 650, 750, 850, 950,1050,1150,
                            1,   1,   1,   1,   1,   1,   1,   1,   1};

  auto e = perspective_embedding(Mat2D(5, 9, matchesData), 1);
  std::cout << find_polynomials(e.getV().back(), e.getD().back(), FISHER, 1) << std::endl;

  return 0;
}

