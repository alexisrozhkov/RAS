#include <iostream>
#include <perspective_embedding.h>

using namespace std;

// Matlab-friendly notation for testing
// matchesData = [1; 2; 3; 4]
// matchesData = [1; 2; 3; 4; 5]
// matchesData = [1, 5; 2, 6; 3, 7; 4, 8]
// matchesData = [1, 6; 2, 7; 3, 8; 4, 9; 5, 10]

int main() {
  double matchesData52[] = {0, 6,
                            2, 0,
                            3, 8,
                            4, 9,
                            5, 10};

  auto embedding = (perspective_embedding(Mat2D(5, 2, matchesData52), 2, false));

  {
    auto V = embedding.getV().back();
    std::cout << V << "\n------------------\n";
  }

  {
    auto D = embedding.getD().back();

    std::cout << D.size() << "x" << D[0].rows << "x" << D[0].cols << std::endl;


    for (int k = 0; k < D[0].cols; k++) {
      for(int j = 0; j < D.size(); j++) {
        for(int i = 0; i < D[0].rows; i++) {
          std::cout << D[j](i, k) << "\t";
        }
        std::cout << "\n";
      }
      std::cout << "\n------------------\n";
    }
  }

  {
    auto H = embedding.getH().back();

    std::cout << H.size() << "x" << H[0].size() << "x" << H[0][0].rows << "x" << H[0][0].cols << std::endl;

    for (int l = 0; l < H[0][0].cols; l++) {
      for (int k = 0; k < H[0][0].rows; k++) {
        for (int j = 0; j < H.size(); j++) {
          for (int i = 0; i < H[0].size(); i++) {
            std::cout << H[j][i](k, l) << "\t";
          }
          std::cout << "\n";
        }
        std::cout << "\n------------------\n";
      }
    }
  }

  return 0;
}

