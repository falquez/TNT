

#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

#include <TNT/configuration/configuration.h>
#include <TNT/configuration/observables.h>
#include <TNT/network/mps/mps.h>
#include <TNT/network/network.h>
#include <TNT/operator/mpo.h>
#include <TNT/operator/observable.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/eigensolver.h>
#include <TNT/tensor/tensor.h>

using NumericalType = double;

int main(int argc, char **argv) {
  using namespace TNT;
  int err = 0;

  auto norm = Tensor::SVDNorm::equal;
  auto tolerance = 1E-09;

  Tensor::Tensor<NumericalType> M1, M2, M3, C1, C2, C3;
  Tensor::Tensor<NumericalType> M({4, 5});
  std::vector<double> m_sv, c_sv;
  M[{0, 0}] = 1.0;
  M[{0, 4}] = 2.0;
  M[{1, 2}] = 3.0;
  M[{3, 1}] = 2.0;

  std::tie(M1, m_sv, M2) = M("a,b").SVD({{"a,c", "c,b"}}, {norm, 3, tolerance});

  std::cout << "M1=" << M1 << std::endl;
  std::cout << "M2=" << M2 << std::endl;

  std::tie(C1, c_sv, C2) = (M1("a,c") * M2("c,b")).SVD({"a,c", "c,b"}, {norm, 3, tolerance});

  std::cout << "C1=" << C1 << std::endl;
  std::cout << "C2=" << C2 << std::endl;

  C3("a,b") = C1("a,c") * C2("c,b");
  M3("a,b") = M1("a,c") * M2("c,b");

  std::cout << M - M3 << std::endl;
  std::cout << M - C3 << std::endl;
  std::cout << C3 - M3 << std::endl;

  return err;
}
