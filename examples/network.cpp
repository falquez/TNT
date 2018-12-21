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

#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include <TNT/configuration/configuration.h>
#include <TNT/configuration/observables.h>
#include <TNT/network/mps/mps.h>
#include <TNT/network/network.h>
#include <TNT/operator/mpo.h>
#include <TNT/operator/observable.h>
#include <TNT/operator/projection.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/eigensolver.h>
#include <TNT/tensor/tensor.h>

using NumericalType = double;
using UInt = unsigned int;

std::string format(unsigned int n, unsigned int w = 4) {
  std::ostringstream str;
  str << std::setw(w) << std::setfill('0') << n;
  return str.str();
}

int main(int argc, char **argv) {
  using namespace TNT;
  int err = 0;

  if (argc != 4) {
    std::cout << argv[0] << " <network size> <operators> <network directory>";
    exit(0);
  }
  const unsigned long L = std::stoul(argv[1]);
  const std::string op_hdf5(argv[2]);
  const std::string network_dir(argv[3]);
  const double tolerance = 1E-16;
  const int maxR = 4;

  std::vector<Tensor::Tensor<NumericalType>> PE(maxR);
  for (int R = 0; R < maxR; R++) {
    std::cout << "Reading Operator " << R << " from " << op_hdf5 << std::endl;
    PE[R] = Tensor::Tensor<NumericalType>(op_hdf5, "/HilbertSpace/Operator/PE", R);

    const unsigned int dimH = PE[R].dimension()[0];

    std::cout << "PE[" << R << "]=" << std::endl;
    for (unsigned int s1 = 0; s1 < dimH; s1++) {
      for (unsigned int s2 = 0; s2 < dimH; s2++) {
        for (unsigned int s3 = 0; s3 < dimH; s3++) {
          for (unsigned int s4 = 0; s4 < dimH; s4++) {
            auto v = PE[R][{s1, s2, s3, s4}];
            if (std::abs(v) > 1E-6)
              std::cout << "[" << s1 << "," << s2 << "," << s3 << "," << s4 << "]=" << v << ",";
          }
        }
      }
    }
    std::cout << std::endl;
  }

  Network::MPS::MPS<NumericalType> A(L);

  for (unsigned int l = 1; l <= L; l++) {
    // Tensor::Tensor<NumericalType> T1, T2;

    std::cout << "Reading tensor " << l << " from " << network_dir + format(l) << std::endl;
    A[l] = Tensor::Tensor<NumericalType>(network_dir + format(l), "/Tensor");

    // T1("a1,a2") = A[l]("s1,a1,a") * A[l].conjugate()("s1,a2,a");
    // T2("a1,a2") = A[l]("s1,a,a1") * A[l].conjugate()("s1,a,a2");

    // T1.tolerance = 1E-8;
    // T2.tolerance = 1E-8;
    // std::cout << "T1[" << l << "]:" << T1 << std::endl;
    // std::cout << "T2[" << l << "]:" << T2 << std::endl;
  }

  for (unsigned int l = 1; l < L / 2; l++) {
    Tensor::Tensor<NumericalType> T;
    const unsigned int r = l + 1;
    const unsigned int mbd = A[l].dimension()[2];
    auto norm = Tensor::SVDNorm::left;

    T("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");

    std::cout << "INFO: Left Decompose T into A[" << l << "]*A[" << r << "]" << std::endl;

    std::tie(A[l], A.SV(l), A[r]) = T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, mbd, tolerance});
  }

  for (unsigned int r = L; r - 1 > L / 2; r--) {
    Tensor::Tensor<NumericalType> T;
    const unsigned int l = r - 1;
    const unsigned int mbd = A[l].dimension()[2];
    auto norm = Tensor::SVDNorm::right;

    T("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");

    std::cout << "INFO: Right Decompose T into A[" << l << "]*A[" << r << "]" << std::endl;

    std::tie(A[l], A.SV(l), A[r]) = T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, mbd, tolerance});
  }

  {
    Tensor::Tensor<NumericalType> T, T2, dT;
    std::vector<Tensor::Tensor<NumericalType>> Ac(maxR);
    std::vector<Tensor::Tensor<NumericalType>> Bc(maxR);
    std::vector<Tensor::Tensor<NumericalType>> PT(maxR);
    std::vector<std::vector<double>> S(maxR);

    const unsigned int l = L / 2;
    const unsigned int r = l + 1;
    // const unsigned int mbd = A[l].dimension()[2];
    const unsigned int dimH = A[l].dimension()[0];
    const unsigned int dimB = A[l].dimension()[1];
    auto norm = Tensor::SVDNorm::equal;

    T("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");

    for (int R = 0; R < maxR; R++) {
      PT[R]("s1,s2,a1,a2") = PE[R]("s1,s2,s1',s2'") * T("s1',s2',a1,a2");
      std::cout << "INFO: Decompose T into A*S*A "
                << "norm=" << norm << " dimB=" << dimB << " tol=" << tolerance << std::endl;

      std::tie(Ac[R], S[R], Bc[R]) = PT[R]("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, dimB, tolerance});

      auto svdname = std::to_string(R) + "_SVD.txt";
      std::cout << "INFO: Writing " << svdname << std::endl;
      std::ofstream ofile(svdname);
      ofile.precision(std::numeric_limits<double>::max_digits10);
      for (unsigned int i = 0; i < S[R].size(); i++) {
        ofile << i << " " << S[R][i] << std::endl;
      }
    }
    T2 = PT[0];
    for (int R = 1; R < maxR; R++) {
      T2 += PT[R];
    }
    dT = T - T2;
    std::cout << "INFO: Decompose T into A[" << l << "]*A[" << r << "]" << std::endl;

    std::cout << "T=" << std::endl;
    for (unsigned int s1 = 0; s1 < dimH; s1++) {
      for (unsigned int s2 = 0; s2 < dimH; s2++) {
        for (unsigned int a1 = 0; a1 < dimB; a1++) {
          for (unsigned int a2 = 0; a2 < dimB; a2++) {
            auto v = T[{s1, s2, a1, a2}];
            if (std::abs(v) > 1E-6)
              std::cout << "[" << s1 << "," << s2 << "," << a1 << "," << a2 << "]=" << v << ",";
          }
        }
      }
    }
    std::cout << std::endl;
    std::cout << "T2=" << std::endl;
    for (unsigned int s1 = 0; s1 < dimH; s1++) {
      for (unsigned int s2 = 0; s2 < dimH; s2++) {
        for (unsigned int a1 = 0; a1 < dimB; a1++) {
          for (unsigned int a2 = 0; a2 < dimB; a2++) {
            auto v = T2[{s1, s2, a1, a2}];
            if (std::abs(v) > 1E-6)
              std::cout << "[" << s1 << "," << s2 << "," << a1 << "," << a2 << "]=" << v << ",";
          }
        }
      }
    }
    std::cout << std::endl;
    std::cout << "dT=" << std::endl;
    for (unsigned int s1 = 0; s1 < dimH; s1++) {
      for (unsigned int s2 = 0; s2 < dimH; s2++) {
        for (unsigned int a1 = 0; a1 < dimB; a1++) {
          for (unsigned int a2 = 0; a2 < dimB; a2++) {
            auto v = dT[{s1, s2, a1, a2}];
            if (std::abs(v) > 1E-6)
              std::cout << "[" << s1 << "," << s2 << "," << a1 << "," << a2 << "]=" << v << ",";
          }
        }
      }
    }
    std::cout << std::endl;
  }
}
