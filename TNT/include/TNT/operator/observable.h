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

#include <TNT/configuration/operator.h>
#include <TNT/tensor/tensor.h>

#include <vector>

namespace TNT::Operator {
  using UInt = unsigned int;
  enum class ObservableType { Site, Correlation };
  template <typename F>
  class Observable {
    ObservableType kind;
    UInt _dimH;
    UInt _length;
    std::string _name;
    std::vector<Tensor::Tensor<F>> O;

  public:
    Observable(){};
    Observable(const std::string &name, const ObservableType &kind,
               const Configuration::Operator<F> &op);
    const Tensor::Tensor<F> &operator[](unsigned int site) const;

    ObservableType type() const { return kind; }
    std::string name() const { return _name; }
  };
} // namespace TNT::Operator

#endif // _TNT_OPERATOR_OBSERVABLE_H
