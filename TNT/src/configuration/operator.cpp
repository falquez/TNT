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

#include <TNT/configuration/operator.h>
#include <TNT/storage/storage.h>

#include <METL/metl.h>
#include <complex>
#include <fstream>
#include <nlohmann/json.hpp>

#include "../parser/parser.h"

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

  template <>
  std::vector<std::vector<std::complex<double>>> read_rows(const std::vector<nlohmann::json> &rows) {
    throw std::invalid_argument("Not Implemented");
  }

  template <typename F>
  std::map<std::string, Operator<F>> Operators(const std::string &config_file) {
    std::map<std::string, Operator<F>> result;

    nlohmann::json j;
    std::ifstream f(config_file);
    f >> j;

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
      // operators.push_back(op);
      result.emplace(name, op);
    }
    return result;
  }
} // namespace TNT::Configuration

template std::map<std::string, TNT::Configuration::Operator<double>>
TNT::Configuration::Operators<double>(const std::string &config_file);

template std::map<std::string, TNT::Configuration::Operator<std::complex<double>>>
TNT::Configuration::Operators<std::complex<double>>(const std::string &config_file);
