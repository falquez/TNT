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
  const auto parameters = config.parameters;

  const auto L = config.network.length;
  const auto results_dir = config.directory("results");
  const auto network_dir = config.directory("network");

  const unsigned int n_max = 10;

  for (const auto [p_i, params] : parameters.iterate()) {

    const TNT::Configuration::Observables<NumericalType> observables(config.config_file, params);
    std::vector<Network::MPS::MPS<NumericalType>> A;
    std::vector<NumericalType> E(n_max);

    for (unsigned int n = 0; n < n_max; n++) {
      const Operator::MPO<NumericalType> W(config, params);

      std::cout << "Calculate W2" << std::endl;
      const auto W2 = W * W;
      std::cout << "Calculate W2c" << std::endl;
      const auto W2c = (W * W).compress();

      std::cout << "W2=" << W2 << std::endl;
      std::cout << "W2c=" << W2c << std::endl;

      const auto output_dir = config.directory("results") + "/" + format(n) + "/" + format(p_i) + "/";
      const auto network_dir = output_dir + config.directory("network") + "/";
      // Create network output directories
      boost::filesystem::create_directories(network_dir);

      // Create new MPS
      A.push_back(Network::MPS::MPS<NumericalType>(config));

      Network::State state(output_dir + "state.json", config.restart);
      if (state.restarted) {
        std::cout << "Restarting MPS A[" << n << "] from iteration " << state.iteration << std::endl;
        for (unsigned int l = 1; l <= L; l++)
          A[n][l] = Tensor::Tensor<NumericalType>(network_dir + format(l), "/Tensor");
      } else {
        std::cout << "Initializing MPS A[" << n << "]" << std::endl;
        A[n].initialize();
        for (unsigned int l = 1; l <= L; l++)
          A[n][l].writeToFile(network_dir + format(l), "/Tensor");
      }

      // Check if already finished
      if (boost::filesystem::exists(output_dir + "result.txt"))
        continue;

      const auto [i_l, i_r, i_dir] = A[n].position(state);

      // Projection Operators
      std::vector<Tensor::TensorScalar<NumericalType>> Pr(n);

      // Right and Left Contractions
      // 1 and L are boundary sites
      std::vector<Tensor::Tensor<NumericalType>> LC(L + 1);
      std::vector<Tensor::Tensor<NumericalType>> RC(L + 1);

      // std::vector<Tensor::Tensor<NumericalType>> LC2(L + 1);
      // std::vector<Tensor::Tensor<NumericalType>> RC2(L + 1);

      // Initialize Right Contractions
      std::cout << "INFO: Initializing Right Contractions" << std::endl;
      RC[L] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
      // RC2[L] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
      for (unsigned int l = L - 1; l >= i_l; l--) {
        RC[l]("b1,a1,a1'") = A[n][l + 1]("s,a1,a2") * W[l + 1]("b1,b2,s,s'") * RC[l + 1]("b2,a2,a2'") *
                             A[n][l + 1].conjugate()("s',a1',a2'");
        // RC2[l]("b1,a1,a1'") = A[n][l + 1]("s,a1,a2") * W2[l + 1]("b1,b2,s,s'") * RC2[l + 1]("b2,a2,a2'") *
        //                      A[n][l + 1].conjugate()("s',a1',a2'");
      }

      // Initialize Left Contractions
      std::cout << "INFO: Initializing Left Contractions" << std::endl;
      LC[1] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
      // LC2[1] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
      for (unsigned int l = 1; l < i_r; l++) {
        LC[l + 1]("b2,a2,a2'") =
            A[n][l]("s,a1,a2") * W[l]("b1,b2,s,s'") * LC[l]("b1,a1,a1'") * A[n][l].conjugate()("s',a1',a2'");
        // LC2[l + 1]("b2,a2,a2'") =
        //    A[n][l]("s,a1,a2") * W2[l]("b1,b2,s,s'") * LC2[l]("b1,a1,a1'") * A[n][l].conjugate()("s',a1',a2'");
      }

      // Start sweep loop
      for (const auto [l, r, dir] : A[n].sweep(state)) {

        // Define Eigensolver for Operator LW*W*W*RW
        Tensor::EigenSolver ES(LC[l]("b1,a1,a1'") * W[l]("b1,b2,s1,s1'") * W[r]("b2,b3,s3,s3'") * RC[r]("b3,a2,a2'"));

        // Calculate Projection Operators
        for (unsigned int n_i = 0; n_i < n; n_i++)
          Pr[n_i] = {A[n_i](A[n], {l, r}), {-E[n_i]}};

        // Optimize A[l]*A[l+1]
        std::cout << "INFO: Optimize A[" << l << "]*A[" << r << "]"
                  << ", tol=" << config.tolerance("eigenvalue") << std::endl;
        auto [ew, T] = ES({{"s1,s3,a1,a2", "s1',s3',a1',a2'"}})
                           .useInitial()
                           .setTolerance(config.tolerance("eigenvalue"))
                           .optimize(A[n][l]("s1,a1,a") * A[n][r]("s3,a,a2"), Pr);

        // @TODO: read max bond dimension from predefined vector, not current dim
        auto nsv = A[n][l]("s1,a1,a").dimension("a");

        auto norm = dir == Network::MPS::Sweep::Direction::Right ? Tensor::SVDNorm::left : Tensor::SVDNorm::right;

        // Perform SVD on T and reassign to A[l], A[r]
        std::cout << "INFO: Decompose T into A[" << l << "]*A[" << r << "]"
                  << " nsv=" << nsv << ", tol=" << config.tolerance("svd") << std::endl;
        // if (nsv < 4)
        std::tie(A[n][l], A[n][r]) =
            T("s1,s3,a1,a2").SVD2({{"s1,a1,a3", "s3,a3,a2"}}, {norm, nsv, config.tolerance("svd")});
        // else
        //  std::tie(A[n][l], A[n][r]) =
        //      T("s1,s3,a1,a2").SVD({{"s1,a1,a3", "s3,a3,a2"}}, A[n][l], A[n][r], {norm, nsv,
        //      config.tolerance("svd")});

        // Store solutions to disk
        A[n][l].writeToFile(network_dir + format(l), "/Tensor");
        A[n][r].writeToFile(network_dir + format(r), "/Tensor");

        // Update left contraction for next iteration
        auto w_lr = dir == Network::MPS::Sweep::Direction::Right ? l : r;
        switch (dir) {
        case Network::MPS::Sweep::Direction::Right:
          LC[r]("b2,a2,a2'") =
              A[n][l]("s,a1,a2") * W[w_lr]("b1,b2,s,s'") * LC[l]("b1,a1,a1'") * A[n][l].conjugate()("s',a1',a2'");
          // LC2[r]("b2,a2,a2'") =
          //    A[n][l]("s,a1,a2") * W2[w_lr]("b1,b2,s,s'") * LC2[l]("b1,a1,a1'") * A[n][l].conjugate()("s',a1',a2'");
          break;
        case Network::MPS::Sweep::Direction::Left:
          RC[l]("b1,a1,a1'") =
              A[n][r]("s,a1,a2") * W[w_lr]("b1,b2,s,s'") * RC[r]("b2,a2,a2'") * A[n][r].conjugate()("s',a1',a2'");
          // RC2[l]("b1,a1,a1'") =
          //    A[n][r]("s,a1,a2") * W2[w_lr]("b1,b2,s,s'") * RC2[r]("b2,a2,a2'") * A[n][r].conjugate()("s',a1',a2'");
          break;
        }

        // std::cout << "INFO: Calculate A[" << n << "](W2) A" << std::endl;
        // T("s1',a1',a'") = A[n][l]("s1,a1,a") * A[n][r]("s3,a,a2") * LC2[l]("b1,a1,a1'") * W2[l]("b1,b2,s1,s1'") *
        //                  W2[r]("b2,b3,s3,s3'") * RC2[r]("b3,a2,a2'") * A[n][r].conjugate()("s3',a',a2'");
        // NumericalType E2A = T("s1',a1',a'") * A[n][l].conjugate()("s1',a1',a'");
        std::cout << "INFO: Calculate A[" << n << "](W2) B" << std::endl;
        double E2 = A[n](W2);
        double NV = params.at("VAR");
        E[n] = ew;
        state.eigenvalue = ew;
        state.variance = (E2 - ew * ew) / (NV * L);

        std::cout << "INFO: n=" << n << " p=" << p_i << " swp=" << state.iteration / L;
        std::cout << " i=" << state.iteration << ", l=" << l << ", r=" << r << ", ";
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        for (const auto &[name, value] : params)
          std::cout << name << "=" << value << ", ";
        std::cout << "ev=" << state.eigenvalue << ", var=" << state.variance;
        // std::cout << ", E2A=" << E2A << ", E2=" << E2;
        std::cout << ", w=" << state.eigenvalue / (2 * L * params.at("x"));
        std::cout << std::endl;

        // Write observables to text file
        for (const auto &[i_o, obs] : observables.iterate()) {
          std::ofstream ofile(output_dir + obs.name + ".txt");
          auto result = A[n](obs);
          for (const auto &r : result) {
            for (const auto &s : r.site)
              ofile << s << " ";
            ofile.precision(std::numeric_limits<double>::max_digits10);
            for (const auto &[n, v] : params)
              ofile << v << " ";
            ofile << state.eigenvalue << " " << state.variance << " ";
            ofile << r.value << std::endl;
          }
          ofile << std::endl;
        }
      }

      // Write observables to text file
      for (const auto &[i_o, obs] : observables.iterate()) {
        std::ofstream ofile(output_dir + obs.name + ".txt");
        auto result = A[n](obs);
        for (const auto &r : result) {
          for (const auto &s : r.site)
            ofile << s << " ";
          ofile.precision(std::numeric_limits<double>::max_digits10);
          for (const auto &[n, v] : params)
            ofile << v << " ";
          ofile << state.eigenvalue << " " << state.variance << " ";
          ofile << r.value << std::endl;
        }
        ofile << std::endl;
      }

      std::ofstream ofile(output_dir + "result.txt");
      ofile << "# D L params.. E var" << std::endl;
      ofile << config.network.dimB << " " << L << " ";
      ofile.precision(std::numeric_limits<double>::max_digits10);
      for (const auto &[n, v] : params)
        ofile << v << " ";
      ofile << state.eigenvalue << " " << state.variance << std::endl;
    }
  }

  return 0;
}
