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

#ifndef _TNT_UTIL_H
#define _TNT_UTIL_H

#include <map>
#include <set>
#include <string>
#include <vector>

namespace TNT::Util {

  std::string format(unsigned int n, unsigned int max);

  template <typename V>
  std::vector<V> select(const std::vector<V> &list, const std::vector<int> &selection);

  template <typename V>
  std::vector<V> selectU(const std::vector<V> &list, const std::vector<unsigned int> &selection);

  template <typename V>
  std::map<V, int> indexed(const std::set<V> &items);

  std::vector<std::string> split(const std::string &str, const std::string &sep);

  std::vector<int> permutation(const std::array<std::string, 2> &str, const std::string &sep = ",");

  std::string concat(const std::vector<std::string> &vec, const std::string &sep = ",");

  bool file_exists(const std::string &filename);

  template <typename V>
  V multiply(const std::vector<V> &dim);

  template <typename V>
  V multiply(const V *array, int size);

  template <typename V>
  std::string stringify(const std::vector<V> &v, const std::string &sep = ",");
} // namespace TNT::Util

#endif //_TNT_UTIL_H
