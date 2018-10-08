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

#ifndef _TNT_CONFIGURATION_H
#define _TNT_CONFIGURATION_H

#include <map>
#include <string>
#include <vector>

#include <TNT/configuration/hamiltonian.h>
#include <TNT/configuration/parameters.h>

namespace TNT::Configuration {
  enum class Topology { MPS };

  struct Tolerance {
    double convergence = 1E-6;
    double eigenvalue = 1E-6;
    double svd = 1E-8;
  };

  struct Network {
    unsigned int length;
    unsigned int dimB;
    Topology topology;
  };

  template <typename F>
  class Configuration {
    std::map<std::string, std::string> directories;
    std::map<std::string, double> tolerances;

  public:
    Configuration(const std::string &config_file);

    std::string directory(const std::string &key) const { return directories.at(key); }

    double tolerance(const std::string &name) const { return tolerances.at(name); };

    Network network;
    Hamiltonian hamiltonian;
    std::map<std::string, Constraint> constraints;
    // std::map<std::string, Operator<F>> operators;

    Parameters parameters;
    const std::string config_file;

    bool restart = false;
  };

} // namespace TNT::Configuration

#endif
