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

#include <TNT/configuration/parameters.h>

#include <METL/metl.h>
#include <nlohmann/json.hpp>

#include <fstream>

namespace TNT::Configuration {

  std::vector<std::map<std::string, double>>
  generate_parameter_list(const std::map<std::string, std::vector<double>> &params) {
    std::vector<std::map<std::string, double>> res;
    std::vector<std::string> keys;
    for (const auto &[k, v] : params)
      keys.push_back(k);
    std::array<std::vector<std::map<std::string, double>>, 2> v;
    for (const auto &p : params.at(keys[0]))
      v[1].push_back({{keys[0], p}});
    for (int i = 1; i < keys.size(); i++) {
      int curr = i % 2;
      int next = (i + 1) % 2;
      v[next].clear();
      for (const auto dict : v[curr]) {
        for (const auto &param : params.at(keys[i])) {
          v[next].push_back(dict);
          v[next].back().emplace(keys[i], param);
        }
      }
    }

    res = v[keys.size() % 2];
    return res;
  }
  Parameters::Parameters(const std::string &config_file) : config_file{config_file} {
    nlohmann::json j;
    std::ifstream f(config_file);
    f >> j;
    // Initialize parameter list
    std::map<std::string, std::string> exprs;
    std::map<std::string, std::vector<double>> params;
    std::map<std::string, nlohmann::json> param_object = j["parameters"];
    for (const auto &[name, obj] : param_object) {
      if (obj.is_array()) {
        std::vector<double> range;
        double start = obj[0];
        double end = obj[1];
        int steps = obj[2];
        double delta = (end - start) / (steps - 1);
        for (int i = 0; i < steps; i++)
          range.push_back(start + delta * i);
        params.emplace(name, range);
      }
      if (obj.is_number()) {
        params.emplace(name, std::vector<double>{obj.get<double>()});
      }
      if (obj.is_string()) {
        exprs.emplace(name, obj.get<std::string>());
      }
    }

    P = generate_parameter_list(params);
    for (auto &m : P) {
      auto compiler = metl::makeCompiler<int, double>();
      metl::setDefaults(compiler);
      for (const auto &[name, value] : m)
        compiler.setConstant(name, value);

      for (const auto &[name, expr] : exprs)
        m.emplace(name, compiler.build<double>(expr)());
    }
  }

  Configuration::Iterator<std::map<std::string, double>> Parameters::iterate() const {
    return Configuration::Iterator(P);
  }

} // namespace TNT::Configuration
