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

#include <complex>

#include <TNT/configuration/observables.h>
#include <TNT/configuration/operator.h>

#include <METL/metl.h>
#include <fstream>
#include <nlohmann/json.hpp>

#include "../parser/parser.h"

namespace TNT::Configuration {

  template <typename F>
  Observables<F>::Observables(const std::string &config_file, const std::map<std::string, double> &P)
      : config_file{config_file} {
    nlohmann::json j;
    std::ifstream f(config_file);

    f >> j;

    std::map<std::string, nlohmann::json> obs_object = j["observables"];

    const auto operators = Operators<F>(config_file);

    const auto parser = Parser::Parser<Tensor::Tensor<F>, F>(config_file, P);

    for (const auto &[name, obs] : obs_object) {
      std::string kind = obs["type"];
      if (kind == "site") {
	std::string op_name = obs["operator"];
	O.push_back(TNT::Operator::Observable<F>(name, TNT::Operator::ObservableType::Site, parser.parse(op_name, 0)));
      }
      if (kind == "correlation") {
        std::string op_name = obs["operator"];
	O.push_back(
	    TNT::Operator::Observable<F>(name, TNT::Operator::ObservableType::Correlation, parser.parse(op_name, 0)));
      }
      if (kind == "shift") {
	O.push_back(TNT::Operator::Observable<F>(name, TNT::Operator::ObservableType::Shift));
        O.back().shift = obs["value"];
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
