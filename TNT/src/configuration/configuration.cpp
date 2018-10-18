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
#include <fstream>
#include <iostream>

#include <TNT/configuration/configuration.h>
#include <TNT/storage/storage.h>

#include <nlohmann/json.hpp>

namespace TNT::Configuration {

  template <typename F>
  Configuration<F>::Configuration(const std::string &config_file) : config_file{config_file}, parameters{config_file} {
    nlohmann::json j;
    std::ifstream f(config_file);
    f >> j;

    restart = j["restart"];
    tolerances = j["tolerance"].get<std::map<std::string, double>>();
    directories = j["directories"].get<std::map<std::string, std::string>>();

    network.dimB = j["network"]["max_bond_dim"];
    network.length = j["network"]["length"];
    // @TODO
    network.topology = Topology::MPS; // j["network"]["topology"]

    // Initialize operators
    // operators = Operators<F>(config_file);

    // Initialize hamiltonian
    auto h_object = j["hamiltonian"];
    hamiltonian.dim = h_object["dim"];
    hamiltonian.n_max = h_object["n_max"];
    auto h_operator = h_object["operators"];
    if (h_operator.find("single") != h_operator.end()) {
      hamiltonian.single_site = h_operator["single"].get<std::string>();
    }

    if (h_operator.find("nearest") != h_operator.end()) {
      std::vector<nlohmann::json> pair_array = h_operator["nearest"];
      for (const auto &pair : pair_array) {
        std::vector<std::string> ops = pair;
        hamiltonian.nearest.push_back({ops[0], ops[1]});
      }
    }

    // Initialize constraints
    std::map<std::string, nlohmann::json> cn_object = j["constraints"];
    for (const auto &[name, obj] : cn_object) {
      Constraint cnt;
      cnt.name = name;
      cnt.expression = obj["operator"];
      if (obj.find("weight") != obj.end()) {
        cnt.weight = obj["weight"];
      }
      if (obj.find("site") != obj.end()) {
        cnt.site = obj["site"];
      }
      if (obj.find("value") != obj.end()) {
        cnt.value = obj["value"];
      }
      // constraints.push_back(cnt);
      constraints.emplace(name, cnt);
    }
    // Initialize operators
    /*std::map<std::string, nlohmann::json> op_object = j["operators"];
    for (const auto &[name, obj] : op_object) {
      Operator<F> op(name);
      if (obj.find("rows") != obj.end()) {
        std::vector<nlohmann::json> rows = obj["rows"];
        op.rows = read_rows<F>(rows);
      } else {
        op.file = obj["file"];
        op.path = obj["path"];

        if (obj.find("sparse") != obj.end())
          op.sparse = true;
        Storage::Storage storage(op.file, Storage::FileMode::ReadOnly);
        Storage::Data::Metadata<unsigned int> size{"size", 0};
        storage.read(op.path, size);
        op.size = size.value;
      }
      // operators.push_back(op);
      operators.emplace(name, op);
    }*/

    // observables = Observables<F>(config_file, operators);
  } // namespace TNT::Configuration
} // namespace TNT::Configuration

template class TNT::Configuration::Configuration<double>;
