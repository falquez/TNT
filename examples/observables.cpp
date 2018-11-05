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

  if (argc != 3) {
    std::cout << argv[0] << " <configuration.json> <directory>";
    exit(0);
  }
  std::string config_file(argv[1]);
  std::string network_dir(argv[2]);

  // Read configuration
  const Configuration::Configuration<NumericalType> config(config_file);
  const auto L = config.network.length;

  /*const auto parameters = config.parameters;

  const auto results_dir = config.directory("results");
  const auto network_dir = config.directory("network");
  const unsigned int n_max = config.hamiltonian.n_max;*/

  // for (const auto [p_i, params] : parameters.iterate()) {

  const TNT::Configuration::Observables<NumericalType> observables(config.config_file);
  Network::MPS::MPS<NumericalType> A(config);
  // std::vector<NumericalType> E(n_max);

  // for (unsigned int n = 0; n < n_max; n++) {
  // const Operator::MPO<NumericalType> W(config, params);
  // std::cout << "INFO: Calculate W2" << std::endl;
  // const auto W2 = W * W;

  // const auto output_dir = config.directory("results") + "/" + format(n) + "/" + format(p_i) + "/";
  // const auto network_dir = output_dir + config.directory("network") + "/";
  // Create network output directories
  // boost::filesystem::create_directories(network_dir);

  // Create new MPS
  // A.push_back(Network::MPS::MPS<NumericalType>(config));

  // Network::State state(output_dir + "state.json", config.restart);
  // if (state.restarted) {
  std::cout << "INFO: Reading MPS" << std::endl;
  for (unsigned int l = 1; l <= L; l++)
    A[l] = Tensor::Tensor<NumericalType>(network_dir + format(l), "/Tensor");

  /*} else {
    std::cout << "INFO: Initializing MPS A[" << n << "]" << std::endl;
    A[n].initialize();
    for (unsigned int l = 1; l <= L; l++)
      A[n][l].writeToFile(network_dir + format(l), "/Tensor");
  }*/

  // Check if already finished
  //    if (boost::filesystem::exists(output_dir + "result.txt"))
  //        continue;

  // const auto [i_l, i_r, i_dir] = A[n].position(state);

  // std::cout << "INFO: Position l=" << i_l << " r=" << i_r << std::endl;
  // Projection Operators
  // std::vector<Tensor::TensorScalar<NumericalType>> Pr(n);

  // Right and Left Contractions
  // 1 and L are boundary sites

  // Write observables to text file
  for (const auto &[i_o, obs] : observables.iterate()) {
    auto obsname = network_dir + obs.name + ".txt";
    std::cout << "INFO: Writing " << obsname << std::endl;
    std::ofstream ofile(obsname);
    auto result = A(obs);
    for (const auto &r : result) {
      for (const auto &s : r.site)
        ofile << s << " ";
      ofile << config.network.dimB << " " << L << " ";
      ofile.precision(std::numeric_limits<double>::max_digits10);
      for (const auto &[n, v] : params)
        ofile << v << " ";
      ofile << state.eigenvalue << " " << state.variance << " ";
      ofile << r.value << std::endl;
    }
    ofile << std::endl;
  }

  std::cout << "INFO: Writing " << output_dir + "result.txt" << std::endl;
  std::ofstream ofile(output_dir + "result.txt");
  ofile << "# D L params.. E var" << std::endl;
  ofile << config.network.dimB << " " << L << " ";
  ofile.precision(std::numeric_limits<double>::max_digits10);
  for (const auto &[n, v] : params)
    ofile << v << " ";
  ofile << state.eigenvalue << " " << state.variance << std::endl;

  return 0;
}
