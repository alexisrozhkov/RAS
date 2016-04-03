// Copyright 2016 Alexey Rozhkov

#include <core/utils/mat_nd.h>
#include <vector>


Mat3D Mat3D_zeros(const int A,
                  const int B,
                  const int C) {
  auto temp = Mat2DArray((uint) A);

  for (int k = 0; k < A; k++) {
    temp[k] = Mat2D::zeros(B, C);
  }

  return temp;
}

Mat4D Mat4D_zeros(const int A,
                  const int B,
                  const int C,
                  const int D) {
  auto temp = Mat3DArray((uint) A);

  for (int k = 0; k < A; k++) {
    temp[k] = Mat3D_zeros(B, C, D);
  }

  return temp;
}

Mat2D Mat2D_clone(const Mat2D &from) {
  return from.clone();
}

Mat3D Mat3D_clone(const Mat3D &from) {
  Mat3D out(from.size());

  for (size_t i = 0; i < out.size(); i++) {
    out[i] = Mat2D_clone(from[i]);
  }

  return out;
}

Mat4D Mat4D_clone(const Mat4D &from) {
  Mat4D out(from.size());

  for (size_t i = 0; i < out.size(); i++) {
    out[i] = Mat3D_clone(from[i]);
  }

  return out;
}

Mat2D filterIdx2(const Mat2D &src, const std::vector<int> &indices) {
  Mat2D out(src.rows, static_cast<int>(indices.size()));

  for (int i = 0; i < static_cast<int>(indices.size()); i++) {
    src.col(indices[i]).copyTo(out.col(i));
  }

  return out;
}

Mat3D filterIdx3(const Mat3D &src, const std::vector<int> &indices) {
  Mat3D out = src;

  for (size_t j = 0; j < src.size(); j++) {
    out[j] = Mat2D(src[j].rows, static_cast<int>(indices.size()));

    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      src[j].col(indices[i]).copyTo(out[j].col(i));
    }
  }

  return out;
}

Mat4D filterIdx4(const Mat4D &src, const std::vector<int> &indices) {
  Mat4D out = src;

  for (size_t k = 0; k < src.size(); k++) {
    for (size_t j = 0; j < src[0].size(); j++) {
      out[k][j] = Mat2D(src[k][j].rows,
                        static_cast<int>(indices.size()));

      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        src[k][j].col(indices[i]).copyTo(out[k][j].col(i));
      }
    }
  }

  return out;
}

Mat2D sliceIdx2(const Mat3D &src, const int idx) {
  Mat2D temp(static_cast<int>(src.size()), src[0].cols);

  for (int j = 0; j < static_cast<int>(src.size()); j++) {
    for (int i = 0; i < src[0].cols; i++) {
      temp(j, i) = src[j](idx, i);
    }
  }

  return temp;
}

Mat2D sliceIdx3(const Mat3D &src, const int idx) {
  Mat2D temp(static_cast<int>(src.size()), src[0].rows);

  for (int j = 0; j < static_cast<int>(src.size()); j++) {
    for (int i = 0; i < src[0].rows; i++) {
      temp(j, i) = src[j](i, idx);
    }
  }

  return temp;
}

Mat2D meanIdx3(const Mat3D &src) {
  Mat2D temp(static_cast<int>(src.size()), src[0].rows);

  for (int j = 0; j < static_cast<int>(src.size()); j++) {
    for (int i = 0; i < src[0].rows; i++) {
      EmbValT sm = 0;
      for (int k = 0; k < src[0].cols; k++) {
        sm += src[j](i, k);
      }

      temp(j, i) = sm/src[0].cols;
    }
  }

  return temp;
}

Mat2D sliceIdx23(const Mat4D &src, const int idx1, const int idx2) {
  Mat2D temp(static_cast<int>(src.size()), src[0][0].cols);

  for (int j = 0; j < static_cast<int>(src.size()); j++) {
    for (int i = 0; i < src[0][0].cols; i++) {
      temp(j, i) = src[j][idx1](idx2, i);
    }
  }

  return temp;
}

// Matlab-like output for easier visual comparison
std::ostream &operator<<(std::ostream &os, Mat3D const &m) {
  os << m.size() << "x" << m[0].rows << "x" << m[0].cols << std::endl;

  for (int k = 0; k < m[0].cols; k++) {
    os << "ans(:,:," << k + 1 << ") = \n\n";

    for (uint j = 0; j < m.size(); j++) {
      for (int i = 0; i < m[0].rows; i++) {
        os << "\t" << m[j](i, k);
      }
      if (j < m.size() - 1) os << "\n";
      else
        os << "\n\n";
    }
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, Mat4D const &m) {
  os << m.size() << "x" << m[0].size() << "x" << m[0][0].rows << "x"
      << m[0][0].cols << std::endl;

  for (int l = 0; l < m[0][0].cols; l++) {
    for (int k = 0; k < m[0][0].rows; k++) {
      os << "ans(:,:," << k + 1 << "," << l + 1 << ") = \n\n";
      for (uint j = 0; j < m.size(); j++) {
        for (uint i = 0; i < m[0].size(); i++) {
          os << "\t" << m[j][i](k, l);
        }

        if (j < m.size() - 1) os << "\n";
        else
          os << "\n\n";
      }
    }
  }

  return os;
}
