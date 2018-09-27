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

#include <iostream>
#include <string>

#include <TNT/operator/sparse/mpo.h>
#include <TNT/tensor/sparse/contraction.h>
#include <TNT/tensor/sparse/tensor.h>

#include "../../parser/parser.h"

namespace TNT::Operator::Sparse {

  /*
   *        3
   *     +--^--+
   *  0->|  W  |->1
   *     +--^--+
   *        2
   */

  template <typename F>
  MPO<F>::MPO(const UInt dimW, const UInt dimH, const UInt length)
      : _dimW{dimW}, _dimH{dimH}, _length{length} {
    W = std::vector<Tensor::Sparse::Tensor<F>>(_length);

    W[0] = Tensor::Sparse::Tensor<F>({1, dimW, dimH, dimH});
    for (unsigned int l = 1; l < _length - 1; l++)
      W[l] = Tensor::Sparse::Tensor<F>({dimW, dimW, dimH, dimH});
    W[_length - 1] = Tensor::Sparse::Tensor<F>({dimW, 1, dimH, dimH});
  }

  template <typename F>
  MPO<F>::MPO(const Configuration::Configuration<F> &conf, const std::map<std::string, double> &P)
      : P{P} {

    auto network = conf.network;
    auto hamiltonian = conf.hamiltonian;
    auto mpo = hamiltonian.mpo.value();

    auto parser = Parser::Parser<Tensor::Sparse::Tensor<F>, F>(conf, P);

    _dimH = hamiltonian.dim;
    _dimW = mpo.dim;
    _length = network.length;

    W = std::vector<Tensor::Sparse::Tensor<F>>(_length);

    W[0] = Tensor::Sparse::Tensor<F>({1, _dimW, _dimH, _dimH});
    for (unsigned int l = 1; l < _length - 1; l++)
      W[l] = Tensor::Sparse::Tensor<F>({_dimW, _dimW, _dimH, _dimH});
    W[_length - 1] = Tensor::Sparse::Tensor<F>({_dimW, 1, _dimH, _dimH});

    for (const auto &block : mpo.blocks) {
      unsigned int row = block.position[0];
      unsigned int col = block.position[1];
      std::string expression;
      if (block.name) {
        expression = *block.name;
      } else if (block.expression) {
        expression = *block.expression;
      }
      if (row == _dimW - 1) {
        Tensor::Sparse::Tensor<F> res = parser.parse(expression, 0);
        for (const auto &[idx, v] : res.elements2())
          W[0] <<= {{0, col, idx[0], idx[1]}, v};
      }
      for (unsigned int l = 1; l < _length - 1; l++) {
        Tensor::Sparse::Tensor<F> res = parser.parse(expression, l);
        for (const auto &[idx, v] : res.elements2())
          W[l] <<= {{row, col, idx[0], idx[1]}, v};
      }
      if (col == 0) {
        Tensor::Sparse::Tensor<F> res = parser.parse(expression, _length);
        for (const auto &[idx, v] : res.elements2())
          W[_length - 1] <<= {{row, 0, idx[0], idx[1]}, v};
      }
    }
  }

  template <typename F>
  MPO<F> MPO<F>::operator*(const MPO<F> &rhs) const {
    assert(_length == rhs._length && _dimH == rhs._dimH);

    MPO<F> result;

    result._dimW = _dimW * rhs._dimW;
    result._dimH = _dimH;
    result._length = _length;

    result.W = std::vector<Tensor::Sparse::Tensor<F>>(_length);

    for (unsigned int l = 0; l < _length; l++) {
      auto P2 = rhs.W[l];

      result.W[l]("b1,b1',b2,b2',a1,a2'") = W[l]("b1,b2,a1,a2") * P2("b1',b2',a2,a2'");
      result.W[l]("b1,b1',b2,b2',a1,a2'").merge("b1,b1'");
      result.W[l]("b1'',b2,b2',a1,a2'").merge("b2,b2'");
    }
    return result;
  }

  template <typename F>
  MPO<F> Identity(const UInt dimW, const UInt dimH, const UInt length) {
    MPO<F> result(dimW, dimH, length);

    for (UInt i = 0; i < dimH; i++) {
      result[1] <<= {{0, 0, i, i}, 1.0};
      result[1] <<= {{0, 1, i, i}, 1.0};
      for (unsigned int l = 2; l < length; l++) {
        result[l] <<= {{0, 0, i, i}, 1.0};
        result[l] <<= {{1, 0, i, i}, 1.0};
        result[l] <<= {{1, 1, i, i}, 1.0};
      }
      result[length] <<= {{0, 0, i, i}, 1.0};
      result[length] <<= {{1, 0, i, i}, 1.0};
    }

    return result;
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const MPO<F> &t) {

    for (int i = 1; i <= t.length(); i++)
      out << "MPO[" << i << "]=" << t[i] << std::endl;

    return out;
  }

} // namespace TNT::Operator::Sparse

template class TNT::Operator::Sparse::MPO<double>;
template class TNT::Operator::Sparse::MPO<std::complex<double>>;

template TNT::Operator::Sparse::MPO<double>
TNT::Operator::Sparse::Identity<double>(const UInt dimW, const UInt dimH, const UInt length);
template TNT::Operator::Sparse::MPO<std::complex<double>>
TNT::Operator::Sparse::Identity<std::complex<double>>(const UInt dimW, const UInt dimH,
                                                      const UInt length);

template std::ostream &TNT::Operator::Sparse::operator<<<double>(std::ostream &,
                                                                 const MPO<double> &);
template std::ostream &TNT::Operator::Sparse::
operator<<<std::complex<double>>(std::ostream &, const MPO<std::complex<double>> &);
