//
// Created by alexey on 28.02.16.
//

#include "ras_types.h"


// Matlab-like output for easier visual comparison

std::ostream &operator<<(std::ostream &os, Mat3D const &m) {
  os << m.size() << "x" << m[0].rows << "x" << m[0].cols << std::endl;

  for (int k = 0; k < m[0].cols; k++) {
    os << "ans(:,:," << k+1 << ") = \n\n";

    for(uint j = 0; j < m.size(); j++) {
      for(int i = 0; i < m[0].rows; i++) {
        os << "\t" << m[j](i, k);
      }
      if(j<m.size()-1) os << "\n";
      else os << "\n\n";
    }
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, Mat4D const &m) {
  os << m.size() << "x" << m[0].size() << "x" << m[0][0].rows << "x" << m[0][0].cols << std::endl;

  for (int l = 0; l < m[0][0].cols; l++) {
    for (int k = 0; k < m[0][0].rows; k++) {
      os << "ans(:,:," << k+1 << "," << l+1 << ") = \n\n";
      for (uint j = 0; j < m.size(); j++) {
        for (uint i = 0; i < m[0].size(); i++) {
          os << "\t" << m[j][i](k, l);
        }

        if(j<m.size()-1) os << "\n";
        else os << "\n\n";
      }
    }
  }

  return os;
}