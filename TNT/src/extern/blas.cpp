
#include "blas.h"
#include <complex>

namespace TNT::BLAS {

  template <>
  double dot(const int &n, const std::array<double *, 2> data) {
    return cblas_ddot(n, data[0], 1, data[1], 1);
  }

  template <>
  std::complex<double> dot(const int &n, const std::array<std::complex<double> *, 2> data) {
    return cblas_zdotu(n, data[0], 1, data[1], 1);
  }
} // namespace TNT::BLAS
