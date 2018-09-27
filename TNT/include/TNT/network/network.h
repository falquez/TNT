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

#ifndef _TNT_NETWORK_H
#define _TNT_NETWORK_H

#include <map>
#include <string>
#include <vector>

namespace TNT::Network {
  using ULong = unsigned long;

  class State {
    std::string filename = "state.json";

  public:
    ULong iteration;
    ULong length;
    double eigenvalue;
    double variance;
    State(const ULong &iteration = 1) : iteration{iteration}, eigenvalue{0}, variance{0} {}

    State(const std::string &config_file, const bool restart = false);

    int writeToFile() const;
  };

} // namespace TNT::Network

#endif // _TNT_NETWORK_H
