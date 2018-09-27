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
#include <iterator>
#include <numeric>

#include <TNT/tensor/vector.h>

namespace TNT::Tensor {
  template <>
  std::optional<double> Vector<double>::operator*(const std::vector<Vector<double>> &lhs) const {
    std::vector<double> s;
    for (const auto &v : lhs)
      if (idx == v.idx)
        s.push_back(_norm * v._norm);
    if (s.empty())
      return {};
    else
      return std::accumulate(std::begin(s), std::end(s), 0.0);
  }

  template <>
  std::optional<std::complex<double>> Vector<std::complex<double>>::
  operator*(const std::vector<Vector<std::complex<double>>> &lhs) const {
    std::vector<std::complex<double>> s;
    for (const auto &v : lhs)
      if (idx == v.idx)
        s.push_back(std::conj(_norm) * v._norm);
    if (s.empty())
      return {};
    else
      return std::accumulate(std::begin(s), std::end(s), std::complex<double>(0, 0));
  }

} // namespace TNT::Tensor

template class TNT::Tensor::Vector<double>;
template class TNT::Tensor::Vector<std::complex<double>>;
