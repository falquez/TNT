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

#include <fstream>
#include <iomanip>
#include <iostream>

#include <TNT/network/network.h>

#include <nlohmann/json.hpp>

#include "../util/util.h"

namespace TNT::Network {

  State::State(const std::string &config_file, const bool restart)
      : filename{config_file}, iteration{1}, eigenvalue{0}, variance{0} {
    nlohmann::json j;

    if (Util::file_exists(filename) && restart) {
      std::ifstream f(filename);
      f >> j;
      iteration = j["iteration"];
      eigenvalue = j["eigenvalue"];
      variance = j["variance"];
      restarted = true;
    } else {
      j["iteration"] = iteration;
      j["eigenvalue"] = eigenvalue;
      j["variance"] = variance;
      std::ofstream f(filename);
      f << std::setw(4) << j << std::endl;
      restarted = false;
    }
  }

  int State::writeToFile() const {
    nlohmann::json j;
    j["iteration"] = iteration;
    j["eigenvalue"] = eigenvalue;
    j["variance"] = variance;

    std::ofstream f(filename);
    f << std::setw(4) << j << std::endl;

    return 0;
  }

} // namespace TNT::Network
