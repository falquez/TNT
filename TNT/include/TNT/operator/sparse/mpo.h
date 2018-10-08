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

#ifndef _TNT_OPERATOR_SPARSE_MPO_H
#define _TNT_OPERATOR_SPARSE_MPO_H

#include <vector>

#include <TNT/configuration/configuration.h>
#include <TNT/operator/mpo.h>
#include <TNT/tensor/sparse/tensor.h>

namespace TNT::Operator::Sparse {

  template <typename F>
  class MPO {
    UInt _dimW;
    UInt _dimH;
    UInt _length;
    std::map<std::string, double> P;
    std::map<std::string, Configuration::Constraint> C;
    std::vector<Tensor::Sparse::Tensor<F>> W;
    MPO(){};

  public:
    MPO(const UInt dimW, const UInt dimH, const UInt length);

    MPO(const Configuration::Configuration<F> &conf, const std::map<std::string, double> &P);
    MPO(const Configuration::Configuration<F> &conf, const std::map<std::string, double> &P,
	const std::map<std::string, Configuration::Constraint> &C);

    // Calculate W("b1,b1',b2,b2',a1,a2'") = W1("b1,b2,a1,a2")W2("b1',b2',a2,a2'")
    // then merge b1''=(b1,b1') b2''=(b2,b2') to W("b1'',b2'',a1,a2'") =
    // W("(b1,b1'),(b2,b2'),a1,a2'")
    MPO<F> operator*(const MPO<F> &rhs) const;

    UInt dimH() const { return _dimH; }

    UInt length() const { return _length; }

    Tensor::Sparse::Tensor<F> &operator[](const UInt &site) { return W[site - 1]; }

    const Tensor::Sparse::Tensor<F> &operator[](const UInt &site) const { return W[site - 1]; }
  };

  template <typename F>
  MPO<F> Identity(const UInt dimW, const UInt dimH, const UInt length);

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const MPO<F> &t);
} // namespace TNT::Operator::Sparse

#endif //_TNT_OPERATOR_SPARSE_MPO_H
