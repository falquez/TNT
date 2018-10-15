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

#ifndef _TNT_TENSOR_SPARSE_CONTRACTION_H
#define _TNT_TENSOR_SPARSE_CONTRACTION_H

#include <map>
#include <vector>

#include <TNT/tensor/sparse/tensor.h>

namespace TNT::Tensor::Sparse {

  template <typename F>
  class Contraction {
  private:
    F dotProduct() const;

  public:
    Contraction(const Tensor<F> &t);

    Contraction<F> &operator*(const Tensor<F> &t);

    operator F() const {
      // Currently support only dot products
      assert(tensors.size() == 2);
      return dotProduct();
    }
    std::vector<std::vector<UInt>> dims;
    std::vector<std::vector<UInt>> strides;
    std::vector<std::string> subs;
    std::vector<const Tensor<F> *> tensors;
    std::map<std::string, UInt> dim_map;
  };

} // namespace TNT::Tensor::Sparse

#endif // _TNT_TENSOR_SPARSE_CONTRACTION_H
