#include <iostream>
#include <perspective_embedding.h>

using namespace std;

// Matlab-friendly notation for testing
// matchesData = [100, 400, 10, 0; 100, 500, 10, 50; 200, 600, 20, 60; 100, 700, 10, 70; 1, 1, 1, 1]

int main() {
  double matchesData[] = {100, 400, 10, 0,
                          100, 500, 10, 50,
                          200, 600, 20, 60,
                          100, 700, 10, 70,
                          1, 1, 1, 1};

  auto tpl = (perspective_embedding(Mat2D(5, 4, matchesData), 2, false, 3));
  auto V = std::get<0>(tpl);
  auto D = std::get<1>(tpl);

  if(1) {
      std::cout << V.back() << "\n------------------\n";
  }

  if(1) {
      for (int i = 0; i < D.back().size(); i++) {
        std::cout << D.back()[i] << "\n------------------\n";
      }
      std::cout << "\n--------------------------------\n";
  }

  return 0;
}

