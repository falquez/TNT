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
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <TNT/configuration/configuration.h>
#include <TNT/network/mps/mps.h>
#include <TNT/network/network.h>
#include <TNT/operator/mpo.h>
#include <TNT/operator/observable.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/eigensolver.h>
#include <TNT/tensor/sparse/contraction.h>
#include <TNT/tensor/sparse/eigensolver.h>
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

  if (argc == 1) {
    std::cout << argv[0] << " <configuration.json>";
    exit(0);
  }
  std::string config_file(argv[1]);

  // Read configuration
  const Configuration::Configuration<NumericalType> config(config_file);
  const auto observables = config.observables;
  const auto parameters = config.parameters;

  const auto L = config.network.length;
  const auto results_dir = config.directory("results");
  const auto network_dir = config.directory("network");

  for (const auto [i_p, params] : parameters.iterate()) {
    // Output directory for this parameter set
    const std::string param_dir = results_dir + "/" + format(i_p);
    // Check if already finished
    if (boost::filesystem::exists(param_dir + "/result.txt"))
      continue;
    // Create output directories
    boost::filesystem::create_directories(param_dir + "/" + network_dir);

    Network::MPS::MPS<NumericalType> A(config);
    const Operator::Sparse::MPO<NumericalType> W(config, params);
    const auto W2 = W * W;

    Network::State state(param_dir + "/state.json");

    const auto [i_l, i_r, i_dir] = A.position(state);

    // Right and Left Contractions
    // 1 and L are boundary sites
    std::vector<Tensor::Tensor<NumericalType>> LC(L + 1);
    std::vector<Tensor::Tensor<NumericalType>> RC(L + 1);

    // Initialize Right Contractions
    std::cout << "Initializing Right Contractions" << std::endl;
    RC[L] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
    for (unsigned int l = L - 1; l >= i_l; l--) {
      Tensor::Tensor DW(W[l + 1]);
      RC[l]("b1,a1,a1'") = A[l + 1]("s,a1,a2") * DW("b1,b2,s,s'") * RC[l + 1]("b2,a2,a2'") *
                           A[l + 1].conjugate()("s',a1',a2'");
    }

    // Initialize Left Contractions
    std::cout << "Initializing Left Contractions" << std::endl;
    LC[1] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
    for (unsigned int l = 1; l < i_r; l++) {
      Tensor::Tensor DW(W[l]);
      LC[l + 1]("b2,a2,a2'") =
          A[l]("s,a1,a2") * DW("b1,b2,s,s'") * LC[l]("b1,a1,a1'") * A[l].conjugate()("s',a1',a2'");
    }

    // Start sweep loop
    for (const auto [l, r, dir] : A.sweep(state)) {
      auto LW = LC[l].sparse();
      auto RW = RC[r].sparse();

      // Define Eigensolver for Operator LW*W*W*RW
      Tensor::Sparse::EigenSolver ES(LW("b1,a1,a1'") * W[l]("b1,b2,s1,s1'") * W[r]("b2,b3,s3,s3'") *
                                     RW("b3,a2,a2'"));

      // Optimize A[l]*A[l+1]
      std::cout << "INFO: Optimize A[" << l << "]*A[" << r << "]"
                << ", acc=" << config.tolerance("eigenvalue") << std::endl;
      auto [ew, T] = ES({{"s1,s3,a1,a2", "s1',s3',a1',a2'"}})
                         .useInitial()
                         .setTolerance(config.tolerance("eigenvalue"))
                         .optimize(A[l]("s1,a1,a") * A[r]("s3,a,a2"));

      // @TODO: read max bond dimension from predefined vector, not current dim
      auto nsv = A[l]("s1,a1,a").dimension("a");

      auto norm = dir == Network::MPS::Sweep::Direction::Right ? Tensor::SVDNorm::left
                                                               : Tensor::SVDNorm::right;
      auto DW = dir == Network::MPS::Sweep::Direction::Right ? Tensor::Tensor(W[l])
                                                             : Tensor::Tensor(W[r]);

      // Perform SVD on T and reassign to A[l], A[r]
      std::cout << "INFO: Decompose T into A[" << l << "]*A[" << r << "]"
                << " nsv=" << nsv << ", acc=" << config.tolerance("svd") << std::endl;
      std::tie(A[l], A[r]) =
          T("s1,s3,a1,a2").SVD({{"s1,a1,a3", "s3,a3,a2"}}, {norm, nsv, config.tolerance("svd")});

      auto E2 = A(W * W);
      state.eigenvalue = ew;
      state.variance = (E2 - ew * ew) / (L * params.at("VAR"));

      std::cout << " ip=" << i_p << " swp=" << state.iteration / L;
      std::cout << " i=" << state.iteration << ", l=" << l << ", r=" << r << ", ";
      std::cout.precision(8);
      for (const auto &[name, value] : params)
        std::cout << name << "=" << value << ", ";
      std::cout << "ev=" << state.eigenvalue << ", var=" << state.variance;
      std::cout.precision(std::numeric_limits<double>::max_digits10);
      std::cout << ", w=" << state.eigenvalue / (2 * L * params.at("x"));
      std::cout << std::endl;

      // Store solutions to disk
      Tensor::writeToFile(A[l], param_dir + "/" + network_dir + "/" + format(l), "/Tensor");
      Tensor::writeToFile(A[r], param_dir + "/" + network_dir + "/" + format(r), "/Tensor");

      // Update left contraction for next iteration
      switch (dir) {
      case Network::MPS::Sweep::Direction::Right:
        LC[r]("b2,a2,a2'") = A[l]("s,a1,a2") * DW("b1,b2,s,s'") * LC[l]("b1,a1,a1'") *
                             A[l].conjugate()("s',a1',a2'");
        break;
      case Network::MPS::Sweep::Direction::Left:
        RC[l]("b1,a1,a1'") = A[r]("s,a1,a2") * DW("b1,b2,s,s'") * RC[r]("b2,a2,a2'") *
                             A[r].conjugate()("s',a1',a2'");
        break;
      }
    }

    for (const auto &[i_o, obs] : observables.iterate()) {
      std::ofstream ofile(param_dir + "/" + obs.name() + ".txt");
      auto result = A(obs);
      for (const auto &r : result) {
        for (const auto &s : r.site)
          ofile << s << " ";
        ofile.precision(std::numeric_limits<double>::max_digits10);
        for (const auto &[n, v] : params)
          ofile << v << " ";
        ofile << r.value << std::endl;
      }
      ofile << std::endl;
    }

    std::ofstream ofile(param_dir + "/result.txt");
    ofile << "# D L params.. E var" << std::endl;
    ofile << config.network.dimB << " " << L << " ";
    ofile.precision(std::numeric_limits<double>::max_digits10);
    for (const auto &[n, v] : params)
      ofile << v << " ";
    ofile << state.eigenvalue << " " << state.variance << std::endl;
  }

  return 0;
}
