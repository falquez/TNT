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

  /*namespace Hamiltonian {
    struct Operator {};
  } // namespace Hamiltonian
  namespace MPO {
    struct Block {
      std::vector<unsigned int> position;
      std::optional<std::string> expression;
      std::optional<std::string> name;
    };

    struct MPO {
      unsigned int dim;
      std::vector<Block> blocks;
    };
  } // namespace MPO
*/
  struct Hamiltonian {
    unsigned int dim;
    std::optional<std::string> single_site = {};
    std::vector<std::array<std::string, 2>> nearest;
  };

  struct Constraint {
    std::string name;
    std::string expression;
    double weight;
    unsigned int site = 0;
  };

} // namespace TNT::Configuration

#endif
