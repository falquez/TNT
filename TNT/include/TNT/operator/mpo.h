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

#ifndef _TNT_OPERATOR_MPO_H
#define _TNT_OPERATOR_MPO_H

#include <TNT/configuration/configuration.h>
#include <TNT/tensor/tensor.h>

#include <vector>

namespace TNT::Operator {
  using UInt = unsigned int;

  /*
   *        3
   *     +--^--+
   *  0->|  W  |->1
   *     +--^--+
   *        2
   */

  template <typename F>
  class MPO {
    UInt _dimW;
    UInt _dimH;
    UInt _length;
    std::vector<UInt> dim;
    std::map<std::string, double> P;
    std::vector<Tensor::Tensor<F>> W;
    MPO(){};

  public:
    MPO(const UInt dimW, const UInt dimH, const UInt length);

    MPO(const Configuration::Configuration<F> &conf, const std::map<std::string, double> &P);

    // Calculate W("b1,b1',b2,b2',a1,a2'") = W1("b1,b2,a1,a2")W2("b1',b2',a2,a2'")
    // then merge b1''=(b1,b1') b2''=(b2,b2') to W("b1'',b2'',a1,a2'") =
    // W("(b1,b1'),(b2,b2'),a1,a2'")
    MPO<F> operator*(const MPO<F> &rhs) const;

    MPO<F> &compress(const UInt &dimW, const double &tolerance = 1E-08);

    UInt dimH() const { return _dimH; }
    UInt dimW() const { return _dimW; }

    UInt length() const { return _length; }

    Tensor::Tensor<F> &operator[](const UInt &site) { return W[site - 1]; }

    const Tensor::Tensor<F> &operator[](const UInt &site) const { return W[site - 1]; }
  };

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const MPO<F> &t);
} // namespace TNT::Operator

#endif //_TNT_OPERATOR_MPO_H
