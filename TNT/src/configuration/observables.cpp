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

#include <TNT/configuration/observables.h>

#include <METL/metl.h>
#include <nlohmann/json.hpp>

#include <fstream>

namespace TNT::Configuration {

  template <typename F>
  Observables<F>::Observables(const std::string &config_file,
                              const std::map<std::string, Operator<F>> &operators)
      : config_file{config_file} {
    nlohmann::json j;
    std::ifstream f(config_file);

    f >> j;

    std::map<std::string, nlohmann::json> obs_object = j["observables"];

    for (const auto &[name, obs] : obs_object) {
      std::string kind = obs["type"];
      TNT::Operator::ObservableType _type;

      if (kind == "site")
        _type = TNT::Operator::ObservableType::Site;
      if (kind == "correlation")
        _type = TNT::Operator::ObservableType::Correlation;

      if (obs.find("operator") != obs.end()) {
        std::string op_name = obs["operator"];
        O.push_back(TNT::Operator::Observable<F>(name, _type, operators.at(op_name)));
      }
    }
  }

  template <typename F>

  Iterator<TNT::Operator::Observable<F>> Observables<F>::iterate() const {
    return Iterator(O);
  }
} // namespace TNT::Configuration

template class TNT::Configuration::Observables<double>;
template class TNT::Configuration::Observables<std::complex<double>>;
