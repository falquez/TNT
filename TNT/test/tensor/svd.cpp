/*
 *  Tensor Network Toolkit
 *  Copyright (C) 2018 Carlos Falquez (falquez@nuberisim.de)
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <catch2/catch.hpp>

#include <TNT/network/mps/mps.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

using NumericalType = double;

/*TEST_CASE("Tensor SVD", "[tensor]") {
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

  std::tie(T1, T2) = T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, nsv, tolerance});

  T3("s1,s2,a1,a2") = T1("s1,a1,a") * T2("s2,a,a2");

  std::cout << "A[3]=" << A[3] << std::endl;
  std::cout << "A[4]=" << A[4] << std::endl;
  std::cout << "T=" << T << std::endl;
  std::cout << "T1=" << T1 << std::endl;
  std::cout << "T2=" << T2 << std::endl;
  std::cout << "T3=" << T3 << std::endl;
  std::cout << "T-T3=" << T - T3 << std::endl;
  REQUIRE(1 == 1);
}*/

TEST_CASE("Tensor SVD 2", "[tensor]") {
  using namespace TNT;

  Network::MPS::MPS<NumericalType> A(10, 3, 4);
  A.initialize();

  // for (int l = 1; l <= A.size(); l++)
  //  std::cout << "A[" << l << "]=" << A[l] << std::endl;

  auto nsv = A[3]("s1,a1,a").dimension("a");
  auto norm = Tensor::SVDNorm::left;
  auto tolerance = 1E-09;

  /*Tensor::Tensor<NumericalType> M1, M2, M3;
  Tensor::Tensor<NumericalType> M({4, 5});
  std::vector<NumericalType> m_sv;
  M[{0, 0}] = 1.0;
  M[{0, 4}] = 2.0;
  M[{1, 2}] = 3.0;
  M[{3, 1}] = 2.0;

  std::tie(M1, m_sv, M2) = M("a,b").SVD({{"a,c", "c,b"}}, {norm, 3, tolerance});

  M3("a,b") = M1("a,c") * M2("c,b");
  std::cout << M - M3 << std::endl;*/

  Tensor::Tensor<NumericalType> T, T1, T2, T3, C1, C2, C3, N1, N2, N3, N4;
  std::vector<NumericalType> t_sv, c_sv;

  T("s1,s2,a1,a2") = A[3]("s1,a1,a") * A[4]("s2,a,a2");
  std::tie(T1, t_sv, T2) = T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, nsv, tolerance});
  std::tie(C1, c_sv, C2) = (A[3]("s1,a1,a") * A[4]("s2,a,a2")).SVD({"s1,a1,a", "s2,a,a2"}, {norm, nsv, tolerance});

  C3("s1,s2,a1,a2") = C1("s1,a1,a") * C2("s2,a,a2");
  T3("s1,s2,a1,a2") = T1("s1,a1,a") * T2("s2,a,a2");
  N1("a1,a1'") = T1("s1,a1,a2") * T1.conjugate()("s1,a1',a2");
  N2("a2,a2'") = T1("s1,a1,a2") * T1.conjugate()("s1,a1,a2'");
  N3("a1,a1'") = T2("s1,a1,a2") * T2.conjugate()("s1,a1',a2");
  N4("a2,a2'") = T2("s1,a1,a2") * T2.conjugate()("s1,a1,a2'");

  std::cout << "A[3]=" << A[3] << std::endl;
  std::cout << "A[4]=" << A[4] << std::endl;

  std::cout << "C1=" << C1 << std::endl;
  std::cout << "C2=" << C2 << std::endl;

  std::cout << "T1=" << T1 << std::endl;
  std::cout << "T2=" << T2 << std::endl;

  std::cout << "T=" << T << std::endl;
  std::cout << "C=" << C3 << std::endl;
  std::cout << "T3-T=" << T3 - T << std::endl;
  std::cout << "C-T=" << C3 - T << std::endl;

  std::cout << "N1=" << N1 << std::endl;
  std::cout << "N2=" << N2 << std::endl;
  std::cout << "N3=" << N3 << std::endl;
  std::cout << "N4=" << N4 << std::endl;

  // std::cout << "A[3]=" << A[3] << std::endl;
  // std::cout << "A[4]=" << A[4] << std::endl;
  // std::cout << "T=" << T << std::endl;
  // std::cout << "T1=" << T1 << std::endl;
  // std::cout << "T2=" << T2 << std::endl;
  // std::cout << "T3=" << T3 << std::endl;
  // std::cout << "T-T3=" << T - T3 << std::endl;
  REQUIRE(1 == 1);
}
