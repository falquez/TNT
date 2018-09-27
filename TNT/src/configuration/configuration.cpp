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

#include <METL/metl.h>
#include <nlohmann/json.hpp>

namespace TNT::Configuration {
  template <typename F>
  std::vector<std::vector<F>> read_rows(const std::vector<nlohmann::json> &rows);

  template <>
  std::vector<std::vector<double>> read_rows(const std::vector<nlohmann::json> &rows) {
    std::vector<std::vector<double>> res;
    for (unsigned int i = 0; i < rows.size(); i++) {
      std::vector<double> row = rows[i];
      res.push_back(row);
    }
    return res;
  }

  template <typename F>
  Configuration<F>::Configuration(const std::string &config_file)
      : config_file{config_file}, parameters{config_file} {
    nlohmann::json j;
    std::ifstream f(config_file);
    f >> j;

    tolerances = j["tolerance"].get<std::map<std::string, double>>();
    directories = j["directories"].get<std::map<std::string, std::string>>();

    network.dimB = j["network"]["max_bond_dim"];
    network.length = j["network"]["length"];
    // @TODO
    network.topology = Topology::MPS; // j["network"]["topology"]

    // Initialize hamiltonian
    auto h_object = j["hamiltonian"];
    hamiltonian.dim = h_object["dim"];
    if (h_object.find("mpo") != h_object.end()) {
      MPO::MPO mpo;
      auto mpo_object = h_object["mpo"];
      std::vector<nlohmann::json> block_array = mpo_object["blocks"];
      mpo.dim = mpo_object["dim"].get<unsigned int>();
      for (const auto &b : block_array) {
        MPO::Block block;
        block.position = b["position"].get<std::vector<unsigned int>>();
        for (int i = 0; i < block.position.size(); i++)
          block.position[i]--;
        if (b.find("operator") != b.end())
          block.name = b["operator"].get<std::string>();
        if (b.find("expression") != b.end())
          block.expression = b["expression"].get<std::string>();
        mpo.blocks.push_back(block);
      }
      hamiltonian.mpo = mpo;
    }

    // Initialize operators
    std::map<std::string, nlohmann::json> op_object = j["operators"];
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
      operators.emplace(name, op);
    }

    observables = Observables<F>(config_file, operators);
  }
} // namespace TNT::Configuration

template class TNT::Configuration::Configuration<double>;
