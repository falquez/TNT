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

#include <TNT/operator/mpo.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

#include "../parser/parser.h"

namespace TNT::Operator {

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
    W = std::vector<Tensor::Tensor<F>>(_length);

    W[0] = Tensor::Tensor<F>({1, dimW, dimH, dimH});
    for (unsigned int l = 1; l < _length - 1; l++)
      W[l] = Tensor::Tensor<F>({dimW, dimW, dimH, dimH});
    W[_length - 1] = Tensor::Tensor<F>({dimW, 1, dimH, dimH});
  }

  template <typename F>
  MPO<F>::MPO(const Configuration::Configuration<F> &conf, const std::map<std::string, double> &P)
      : P{P} {

    auto network = conf.network;
    auto hamiltonian = conf.hamiltonian;
    auto mpo = hamiltonian.mpo.value();

    auto parser = Parser::Parser<Tensor::Tensor<F>, F>(conf, P);

    _dimH = hamiltonian.dim;
    _dimW = mpo.dim;
    _length = network.length;

    W = std::vector<Tensor::Tensor<F>>(_length);

    W[0] = Tensor::Tensor<F>({1, _dimW, _dimH, _dimH});
    for (unsigned int l = 1; l < _length - 1; l++)
      W[l] = Tensor::Tensor<F>({_dimW, _dimW, _dimH, _dimH});
    W[_length - 1] = Tensor::Tensor<F>({_dimW, 1, _dimH, _dimH});

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
        Tensor::Tensor<F> res = parser.parse(expression, 0);
        for (unsigned int i = 0; i < _dimH; i++)
          for (unsigned int j = 0; j < _dimH; j++)
            W[0][{0, col, i, j}] = res[{i, j}];
      }
      for (unsigned int l = 1; l < _length - 1; l++) {
        Tensor::Tensor<F> res = parser.parse(expression, l);
        for (unsigned int i = 0; i < _dimH; i++)
          for (unsigned int j = 0; j < _dimH; j++)
            W[l][{row, col, i, j}] = res[{i, j}];
      }
      if (col == 0) {
        Tensor::Tensor<F> res = parser.parse(expression, _length);
        for (unsigned int i = 0; i < _dimH; i++)
          for (unsigned int j = 0; j < _dimH; j++)
            W[_length - 1][{row, 0, i, j}] = res[{i, j}];
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

    result.W = std::vector<Tensor::Tensor<F>>(_length);

    for (unsigned int l = 0; l < _length; l++) {
      auto P2 = rhs.W[l];

      result.W[l]("b1,b1',b2,b2',a1,a2'") = W[l]("b1,b2,a1,a2") * P2("b1',b2',a2,a2'");
      result.W[l]("b1,b1',b2,b2',a1,a2'").merge("b1,b1'");
      result.W[l]("b1'',b2,b2',a1,a2'").merge("b2,b2'");
    }
    return result;
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const MPO<F> &t) {

    for (UInt i = 1; i <= t.length(); i++)
      out << "MPO[" << i << "]=" << t[i] << std::endl;

    return out;
  }
} // namespace TNT::Operator

template class TNT::Operator::MPO<double>;
template class TNT::Operator::MPO<std::complex<double>>;

template std::ostream &TNT::Operator::operator<<<double>(std::ostream &, const MPO<double> &);
template std::ostream &TNT::Operator::
operator<<<std::complex<double>>(std::ostream &, const MPO<std::complex<double>> &);
