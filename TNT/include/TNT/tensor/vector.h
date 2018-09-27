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

#ifndef _TNT_TENSOR_VECTOR_H
#define _TNT_TENSOR_VECTOR_H

#include <array>
#include <optional>
#include <tuple>
#include <vector>

namespace TNT::Tensor {

  template <typename F>
  class Vector {
    std::vector<unsigned int> idx;
    const F _norm;

  public:
    Vector(const std::vector<unsigned int> &idx, const F norm = 1.0) : idx(idx), _norm(norm) {}

    // contract <v| \sum_i |lhs_i>
    std::optional<F> operator*(const std::vector<Vector<F>> &lhs) const;

    std::vector<unsigned int> indices() const { return idx; }

    unsigned int operator[](const unsigned int i) const { return idx[i]; }

    F norm() const { return _norm; }
    unsigned int rank() const { return idx.size(); }
  };

  template <typename V>
  class HilbertSpace {
  public:
    virtual const V &operator[](const unsigned int i) const = 0;
    virtual unsigned int operator()(const V &vec) const = 0;
    virtual unsigned int dimension() const = 0;
  };

  template <typename V, typename F>
  class VectorOperator {
  protected:
    const HilbertSpace<V> &space;

  public:
    VectorOperator(const Tensor::HilbertSpace<V> &space) : space{space} {}
    virtual std::vector<Tensor::Vector<F>> operator()(const Tensor::Vector<F> &t) const = 0;
    virtual std::string name() const = 0;
  };

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Vector<F> &t);
} // namespace TNT::Tensor

#endif // _TNT_TENSOR_VECTOR_H
