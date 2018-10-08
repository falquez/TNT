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

#ifndef _TNT_PARSER_PARSER_H
#define _TNT_PARSER_PARSER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <TNT/configuration/configuration.h>

namespace TNT::Parser {

  template <typename T, typename F>
  class Parser {

    std::map<std::string, T> O;
    std::map<std::string, double> P;

  public:
    Parser(const std::map<std::string, Configuration::Operator<F>> &operators, const std::map<std::string, double> &P);

    T parse(const std::string &s, int pos = 0) const;

    unsigned int dimH;
  };

} // namespace TNT::Parser

#endif
