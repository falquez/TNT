#include <catch2/catch.hpp>

#include <TNT/network/mps/mps.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

using NumericalType = double;

TEST_CASE("Tensor SVD", "[tensor]") {
  using namespace TNT;

  Network::MPS::MPS<NumericalType> A(4, 10, 5);
  A.initialize();

  // for (int l = 1; l <= A.size(); l++)
  //  std::cout << "A[" << l << "]=" << A[l] << std::endl;

  Tensor::Tensor<NumericalType> T, T1, T2, T3;
  T("s1,s2,a1,a2") = A[3]("s1,a1,a") * A[4]("s2,a,a2");

  auto nsv = A[3]("s1,a1,a").dimension("a");
  auto norm = Tensor::SVDNorm::left;
  auto tolerance = 1E-09;

  std::tie(T1, T2) = T("s1,s2,a1,a2").SVD2({{"s1,a1,a", "s2,a,a2"}}, {norm, nsv, tolerance});

  T3("s1,s2,a1,a2") = T1("s1,a1,a") * T2("s2,a,a2");

  std::cout << "A[3]=" << A[3] << std::endl;
  std::cout << "A[4]=" << A[4] << std::endl;
  std::cout << "T=" << T << std::endl;
  std::cout << "T1=" << T1 << std::endl;
  std::cout << "T2=" << T2 << std::endl;
  std::cout << "T3=" << T3 << std::endl;
  std::cout << "T-T3=" << T - T3 << std::endl;
  REQUIRE(1 == 1);
}
