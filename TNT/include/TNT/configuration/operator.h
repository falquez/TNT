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

#ifndef _TNT_CONFIGURATION_OPERATOR_H
#define _TNT_CONFIGURATION_OPERATOR_H

#include <map>
#include <string>
#include <vector>

namespace TNT::Configuration {

  template <typename F>
  struct Operator {
    std::vector<std::vector<F>> rows;
    std::string name;
    std::string file;
    std::string path;
    unsigned int size = 1;
    bool sparse = false;
    Operator(const std::string &name) : name{name} {}
  };

  template <typename F>
  std::map<std::string, Operator<F>> Operators(const std::string &config_file);

} // namespace TNT::Configuration

#endif
