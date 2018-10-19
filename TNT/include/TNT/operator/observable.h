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

#ifndef _TNT_OPERATOR_OBSERVABLE_H
#define _TNT_OPERATOR_OBSERVABLE_H

//#include <TNT/configuration/operator.h>
#include <vector>

#include <TNT/tensor/tensor.h>

namespace TNT::Operator {
  using UInt = unsigned int;
  enum class ObservableType { Site, Correlation, Shift };

  template <typename F>
  class Observable {
    ObservableType _kind;
    // UInt _dimH;
    // UInt _length;

    std::vector<Tensor::Tensor<F>> O;

  public:
    Observable(){};
    Observable(const std::string &name, const ObservableType &kind) : _kind{kind}, name{name} {}
    Observable(const std::string &name, const ObservableType &kind, Tensor::Tensor<F> &&O)
	: _kind{kind}, name{name}, O{O} {};
    Observable(const std::string &name, const ObservableType &kind, std::vector<Tensor::Tensor<F>> &&O)
	: _kind{kind}, name{name}, O{O} {};
    const Tensor::Tensor<F> &operator[](unsigned int site) const;

    ObservableType kind() const { return _kind; }

    std::string name;
    UInt shift;
  };
} // namespace TNT::Operator

#endif // _TNT_OPERATOR_OBSERVABLE_H
