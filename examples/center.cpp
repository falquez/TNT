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
#include <TNT/configuration/operator.h>
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

int write_observable(const TNT::Network::MPS::MPS<NumericalType> &A,
                     const TNT::Operator::Observable<NumericalType> &obs, const std::string &obsname,
                     std::map<std::string, double> parameters, unsigned int L, unsigned int dimB) {
  int rc = 0;
  std::ofstream ofile(obsname);
  auto result = A(obs);
  for (const auto &r : result) {
    ofile << dimB << " " << L << " ";
    ofile.precision(std::numeric_limits<double>::max_digits10);
    for (const auto &[n, v] : parameters)
      ofile << v << " ";
    for (const auto &s : r.site)
      ofile << s << " ";
    ofile.precision(std::numeric_limits<double>::max_digits10);
    ofile << r.value << std::endl;
  }
  ofile << std::endl;
  return rc;
}

int calculate_variance(const TNT::Network::MPS::MPS<NumericalType> &A, const TNT::Operator::MPO<NumericalType> &W,
                       const TNT::Operator::MPO<NumericalType> &W2, const bool calculate_var = true) {
  int rc = 0;
  double E1, E2;

  std::cout << "INFO: Calculate A(W)" << std::endl;
  E1 = A(W);

  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cout << "INFO: ev=" << E1 << std::endl;

  std::cout << "INFO: Calculate A(W2)" << std::endl;
  if (calculate_var)
    E2 = A(W2);
  else
    E2 = 0.0;

  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cout << "INFO: var=" << E2 << std::endl;

  double eigenvalue = E1;
  double variance = (E2 - E1 * E1) / (E1 * E1);

  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cout << "INFO: ev=" << eigenvalue << " var=" << variance << std::endl;
  return rc;
}

int main(int argc, char **argv) {
  using namespace TNT;
  int err = 0;

  if (argc != 2) {
    std::cout << argv[0] << " <configuration.json>";
    exit(0);
  }
  const std::string config_file(argv[1]);

  // Read configuration
  const Configuration::Configuration<NumericalType> config(config_file);

  const auto parameters = config.parameters;
  const auto L = config.network.length;
  const auto n_max = config.hamiltonian.n_max;

  const auto results_dir = config.directory("results");
  const auto network_dir = config.directory("network");

  const auto operators = TNT::Configuration::Operators<double>(config_file);
  const unsigned int maxR = operators.at("PE").size;

  std::vector<Tensor::Tensor<NumericalType>> PE(maxR);
  for (int R = 0; R < maxR; R++) {
    std::cout << "Reading Operator " << R << " from " << operators.at("PE").file << std::endl;
    PE[R] = Tensor::Tensor<NumericalType>(operators.at("PE").file, operators.at("PE").path, R);
  }

  for (const auto [p_i, params] : parameters.iterate()) {
    const unsigned int n = 0;
    const auto output_dir = config.directory("results") + "/" + format(n) + "/" + format(p_i) + "/";
    const auto network_dir = output_dir + config.directory("network") + "/";
    const auto centered_dir = network_dir + "centered/";
    boost::filesystem::create_directories(centered_dir);

    const Configuration::Observables<NumericalType> observables(config.config_file, params);

    if (!boost::filesystem::exists(output_dir + "result.txt") ||
        boost::filesystem::exists(output_dir + "energy_center.txt"))
      continue;

    std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Reading W" << std::endl;
    const Operator::MPO<NumericalType> W(config_file, config.hamiltonian.mpo, params);

    std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Calculate W2" << std::endl;
    const auto W2 = W * W;

    // Read MPS
    Network::MPS::MPS<NumericalType> A(config);
    for (unsigned int l = 1; l <= L; l++)
      A[l] = Tensor::Tensor<NumericalType>(network_dir + format(l), "/Tensor");

    std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i
              << " Calculate variance: " << std::endl;
    calculate_variance(A, W, W2);

    for (unsigned int l = 1; l < L / 2; l++) {
      Tensor::Tensor<NumericalType> T;
      const unsigned int r = l + 1;
      const unsigned int mbd = A[l].dimension()[2];
      auto norm = Tensor::SVDNorm::left;

      T("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");

      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Left Decompose T into A[" << l
                << "]*A[" << r << "]";
      std::cout << " norm=" << norm << " mbd=" << mbd << " tol=" << config.tolerance("eigenvalue") << std::endl;

      std::tie(A[l], A.SV(l), A[r]) =
          T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, mbd, config.tolerance("eigenvalue")});

      Tensor::Tensor<NumericalType> T2, dT;
      T2("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");
      dT = T - T2;
      std::cout << "INFO: dT=" << dT.norm2() << std::endl;

      std::cout << "INFO: l=" << l << " r=" << r << " Calculate energy: " << std::endl;
      calculate_variance(A, W, W2, false);
    }

    std::cout << "INFO: Calculate variance: " << std::endl;
    calculate_variance(A, W, W2);

    for (unsigned int r = L; r - 1 > L / 2; r--) {
      Tensor::Tensor<NumericalType> T;
      const unsigned int l = r - 1;
      const unsigned int mbd = A[l].dimension()[2];
      auto norm = Tensor::SVDNorm::right;

      T("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");

      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Right Decompose T into A[" << l
                << "]*A[" << r << "]";
      std::cout << " norm=" << norm << " mbd=" << mbd << " tol=" << config.tolerance("eigenvalue") << std::endl;

      std::tie(A[l], A.SV(l), A[r]) =
          T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, mbd, config.tolerance("eigenvalue")});

      Tensor::Tensor<NumericalType> T2, dT;
      T2("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");
      dT = T - T2;
      std::cout << "INFO: dT=" << dT.norm2() << std::endl;

      std::cout << "INFO: l=" << l << " r=" << r << " Calculate energy: " << std::endl;
      calculate_variance(A, W, W2, false);
    }

    std::cout << "INFO: Calculate variance: " << std::endl;
    calculate_variance(A, W, W2);

    {
      Tensor::Tensor<NumericalType> T;
      std::vector<Tensor::Tensor<NumericalType>> Ac(maxR);
      std::vector<Tensor::Tensor<NumericalType>> Bc(maxR);
      std::vector<Tensor::Tensor<NumericalType>> PT(maxR);
      std::vector<std::vector<double>> S(maxR);

      const unsigned int l = L / 2;
      const unsigned int r = l + 1;

      T("s1,s2,a1,a2") = A[l]("s1,a1,a") * A[r]("s2,a,a2");

      const unsigned int dimH = A[l].dimension()[0];
      const unsigned int dimB = A[l].dimension()[1];
      auto norm = Tensor::SVDNorm::equal;

      std::ofstream ofile(output_dir + "entropy_center.txt");
      ofile << "# D L params.. R i lambda" << std::endl;
      for (int R = 0; R < maxR; R++) {
        PT[R]("s1,s2,a1,a2") = PE[R]("s1,s2,s1',s2'") * T("s1',s2',a1,a2");
        std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Decompose T into A*S*A "
                  << "norm=" << norm << " dimB=" << dimB << " tol=" << config.tolerance("eigenvalue") << std::endl;

        std::tie(Ac[R], S[R], Bc[R]) =
            PT[R]("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, dimB, config.tolerance("eigenvalue")});

        for (unsigned int i = 0; i < S[R].size(); i++) {
          ofile << config.network.dimB << " " << L << " ";
          ofile.precision(std::numeric_limits<double>::max_digits10);
          for (const auto &[n, v] : params)
            ofile << v << " ";
          ofile << R << " " << i << " " << S[R][i] << std::endl;
        }
      }
      ofile << "#" << std::endl;

      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Decompose T into A[" << l
                << "]*A[" << r << "]" << std::endl;
      std::tie(A[l], A.SV(l), A[r]) =
          T("s1,s2,a1,a2").SVD({{"s1,a1,a", "s2,a,a2"}}, {norm, dimB, config.tolerance("eigenvalue")});

      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Writing "
                << output_dir + "entropy2_center.txt" << std::endl;
      std::ofstream ofile2(output_dir + "entropy2_center.txt");
      ofile2.precision(std::numeric_limits<double>::max_digits10);
      const auto svs = A.SV(l);
      for (unsigned int i = 0; i < svs.size(); i++) {
        ofile2 << i << " " << svs[i] << std::endl;
      }
    }
    for (unsigned int l = 1; l <= L; l++) {
      A[l].writeToFile(centered_dir + format(l), "/Tensor");
    }

    {
      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Calculate A(W)" << std::endl;
      double E1 = A(W);
      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Calculate A(W2)" << std::endl;
      double E2 = A(W2);

      double eigenvalue = E1;
      double variance = (E2 - E1 * E1) / (E1 * E1);

      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " ";
      std::cout.precision(std::numeric_limits<double>::max_digits10);
      for (const auto &[name, value] : params)
        std::cout << name << "=" << value << " ";
      std::cout << "ev=" << eigenvalue << " var=" << variance;
      std::cout << std::endl;

      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Writing "
                << output_dir + "energy_center.txt" << std::endl;
      std::ofstream ofile(output_dir + "energy_center.txt");
      ofile << "# D L params.. E var" << std::endl;
      ofile << config.network.dimB << " " << L << " ";
      ofile.precision(std::numeric_limits<double>::max_digits10);
      for (const auto &[n, v] : params)
        ofile << v << " ";
      ofile << eigenvalue << " " << variance << std::endl;
    }

    // Write observables to text file
    for (const auto &[i_o, obs] : observables.iterate()) {
      std::cout << "INFO: D=" << config.network.dimB << " L=" << L << " p=" << p_i << " Writing "
                << output_dir + obs.name + "_center.txt" << std::endl;
      int rc = write_observable(A, obs, output_dir + obs.name + "_center.txt", params, L, config.network.dimB);
    }
  }
}
