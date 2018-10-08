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

#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>
#include <sys/stat.h>

#include "util.h"
#include <boost/algorithm/string.hpp>

namespace TNT::Util {

  std::string format(unsigned int n, unsigned int max) {
    int w = std::lround(std::ceil(std::log10(max))) + 1;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(w) << n;
    return oss.str();
  }

  template <typename V>
  std::vector<V> select(const std::vector<V> &list, const std::vector<int> &selection) {
    std::vector<V> result;
    for (unsigned int i = 0; i < selection.size(); i++)
      result.push_back(list[selection[i]]);
    return result;
  }

  template <typename V>
  std::vector<V> selectU(const std::vector<V> &list, const std::vector<unsigned int> &selection) {
    std::vector<V> result;
    for (unsigned int i = 0; i < selection.size(); i++)
      result.push_back(list[selection[i]]);
    return result;
  }

  template <typename V>
  std::map<V, int> indexed(const std::set<V> &items) {
    std::map<V, int> result;
    int idx = 0;
    for (const auto &item : items)
      result.emplace(item, idx++);
    return result;
  }

  std::vector<std::string> split(const std::string &str, const std::string &sep) {
    std::vector<std::string> result;

    boost::algorithm::split(result, str, boost::algorithm::is_any_of(sep));

    return result;
  }

  std::string concat(const std::vector<std::string> &vec, const std::string &sep) {
    std::string result;
    for (const auto &str : vec)
      result = result + str + sep;
    return result.substr(0, result.size() - sep.size());
  }

  std::vector<int> permutation(const std::array<std::string, 2> &str, const std::string &sep) {
    std::vector<int> trs;
    std::array<std::vector<std::string>, 2> idx;
    std::map<std::string, int> idx_map;

    for (int n = 0; n < 2; n++)
      idx[n] = split(str[n], sep);

    for (int i = 0; i < idx[0].size(); i++)
      idx_map.emplace(idx[0][i], i);

    for (const auto &c : idx[1])
      trs.push_back(idx_map.at(c));

    return trs;
  }

  template <>
  int multiply(const std::vector<int> &vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<>());
  }
  template <>
  unsigned int multiply(const std::vector<unsigned int> &vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 1U, std::multiplies<unsigned int>());
  }

  template <>
  unsigned long long multiply(const std::vector<unsigned long long> &vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 1ULL, std::multiplies<unsigned int>());
  }

  template <typename V>
  V sum(const std::vector<V> &vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 0);
  }

  template <typename V>
  V multiply(const V *array, int size) {
    std::vector<V> dim;
    dim.assign(array, array + size);
    return multiply<V>(dim);
  }

  template <typename V>
  std::string stringify(const std::vector<V> &v, const std::string &sep) {
    std::string result;

    for (const auto &elem : v) {
      result = result + std::to_string(elem) + sep;
    }

    return result.substr(0, result.size() - sep.size());
  }

  bool file_exists(const std::string &filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
  }
} // namespace TNT::Util

template std::map<std::vector<int>, int> TNT::Util::indexed<std::vector<int>>(const std::set<std::vector<int>> &);

template std::vector<int> TNT::Util::select<int>(const std::vector<int> &, const std::vector<int> &);
template std::vector<unsigned int> TNT::Util::selectU<unsigned int>(const std::vector<unsigned int> &,
								    const std::vector<unsigned int> &);
template std::vector<int> TNT::Util::selectU<int>(const std::vector<int> &, const std::vector<unsigned int> &);

template unsigned int TNT::Util::multiply<unsigned int>(unsigned int const *, int);
template int TNT::Util::multiply<int>(int const *, int);

template std::string TNT::Util::stringify<int>(const std::vector<int> &, const std::string &);
template std::string TNT::Util::stringify<unsigned int>(const std::vector<unsigned int> &, const std::string &);
