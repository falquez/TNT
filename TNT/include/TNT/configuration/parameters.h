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

#ifndef _TNT_CONFIGURATION_PARAMETERS_H
#define _TNT_CONFIGURATION_PARAMETERS_H

#include <TNT/configuration/iterator.h>

#include <map>
#include <tuple>
#include <vector>

namespace TNT::Configuration {

  class Parameters {
    const std::string config_file;
    std::vector<std::map<std::string, double>> P;

  public:
    Parameters(){};
    Parameters(const std::string &config_file);
    unsigned int size() const { return P.size(); }
    Iterator<std::map<std::string, double>> iterate() const;
  };
} // namespace TNT::Configuration

#endif
