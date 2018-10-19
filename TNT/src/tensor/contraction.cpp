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

#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>

#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

#include "../extern/blas.h"
#include "../util/util.h"

namespace TNT::Tensor {

  template <typename F>
  Contraction<F>::Contraction(const Tensor<F> &t) : dims{t.dim}, subs{t.sub}, data{t.data.get()}, dim_map{} {

    std::vector<std::string> idx = Util::split(subs[0], ",");

    for (UInt i = 0; i < dims[0].size(); i++)
      dim_map.emplace(idx[i], dims[0][i]);
  }

  template <typename F>
  Contraction<F> &Contraction<F>::operator*(const Tensor<F> &t) {

    data.push_back(t.data.get());
    subs.push_back(t.sub);
    dims.push_back(t.dim);
    // strides.push_back(t.stride);
    // totalDim.push_back(t.totalDim);

    std::vector<std::string> idx = Util::split(subs.back(), ",");
    for (UInt i = 0; i < dims.back().size(); i++)
      dim_map.emplace(idx[i], dims.back()[i]);

    return *this;
  }

  template <typename F>
  Contraction<F>::operator F() const {
    assert(data.size() == 2);

    auto totalDim = Util::multiply(dims[0]);

    return TNT::BLAS::dot<F>(totalDim, {data[0], data[1]});
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Contraction<F> &T) {
    out << "Contraction: (Not Implemented)\n";
    return out;
  }
} // namespace TNT::Tensor

template class TNT::Tensor::Contraction<double>;
template class TNT::Tensor::Contraction<std::complex<double>>;

template std::ostream &TNT::Tensor::operator<<<double>(std::ostream &out, const Contraction<double> &T);
template std::ostream &TNT::Tensor::operator<<<std::complex<double>>(std::ostream &out,
								     const Contraction<std::complex<double>> &T);
