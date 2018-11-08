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

#ifndef _TNT_CONFIGURATION_HAMILTONIAN_H
#define _TNT_CONFIGURATION_HAMILTONIAN_H

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace TNT::Configuration {
  enum class BoundaryCondition { Open, Periodic };

  struct MPO {
    unsigned int length;
    std::optional<std::string> single_site = {};
    std::vector<std::array<std::string, 2>> nearest;
  };

  struct Hamiltonian {
    unsigned int dim;
    unsigned int n_max;
    MPO mpo;
    MPO projection;
  };

  struct Constraint {
    std::string name;
    std::string expression;
    double weight = 0.0;
    double value = 0.0;
    unsigned int site = 0;
  };

} // namespace TNT::Configuration

#endif
