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

#ifndef _TNT_TENSOR_EIGENSOLVER_H
#define _TNT_TENSOR_EIGENSOLVER_H

#include <array>
#include <tuple>

#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

namespace TNT::Tensor {

  template <typename F>
  class EigenSolver {
    Contraction<F> seq;
    std::array<std::string, 2> sub;
    double tolerance;
    std::vector<double> targets;
    bool _useInitial;

  public:
    EigenSolver(const Contraction<F> &seq, const double tolerance = 10E-9)
	: seq{seq}, sub{}, tolerance{tolerance}, targets{}, _useInitial{false} {}

    EigenSolver<F> &setTolerance(const double &tol) {
      tolerance = tol;
      return *this;
    }

    EigenSolver<F> &useInitial(const bool init = true) {
      _useInitial = init;
      return *this;
    }

    EigenSolver<F> &setTargets(const std::vector<double> &target) {
      targets = target;
      return *this;
    }

    EigenSolver<F> &operator()(const std::string &sub1, const std::string &sub2);

    EigenSolver<F> &operator()(const std::array<std::string, 2> &sub1);

    std::tuple<F, Tensor<F>> optimize(const Tensor<F> &t1const, const std::vector<Tensor<F>> &P = {},
				      const std::vector<Tensor<F>> &X = {}) const;
    std::tuple<F, Tensor<F>> optimize(const Contraction<F> &seqconst, const std::vector<Tensor<F>> &P = {},
				      const std::vector<Tensor<F>> &X = {}) const;
  };
} // namespace TNT::Tensor

#endif // _TNT_TENSOR_EIGENSOLVER_H
