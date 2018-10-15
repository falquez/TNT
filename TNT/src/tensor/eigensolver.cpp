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

#include <TNT/tensor/eigensolver.h>

#include "../algebra/algebra.h"
#include "../util/util.h"

namespace TNT::Tensor {
  template <typename F>
  std::tuple<F, Tensor<F>> EigenSolver<F>::optimize(const Tensor<F> &t1, const std::vector<TensorScalar<F>> &P,
                                                    const std::vector<TensorScalar<F>> &X) const {
    int err = 0;

    double ev;
    std::unique_ptr<F[]> buff;

    auto idx = Util::split(sub[0], ",");
    std::vector<UInt> vec_dim(idx.size());
    for (unsigned int i = 0; i < vec_dim.size(); i++)
      vec_dim[i] = seq.dim_map.at(idx[i]);

    Tensor<F> T(vec_dim);

    int initSize = 0;
    buff = std::make_unique<F[]>(T.totalDim);

    if (_useInitial) {
      initSize = 1;
      t1.writeTo(buff.get());
    }

    err = Algebra::tensorEigen(&ev, buff.get(), sub, seq, P, X,
                               Algebra::Options(1, tolerance, initSize, Algebra::Target::smallest, {}));

    T.data = std::move(buff);

    return std::make_tuple(ev, std::move(T));
  }

  template <typename F>
  std::tuple<F, Tensor<F>> EigenSolver<F>::optimize(const Contraction<F> &seq1, const std::vector<TensorScalar<F>> &P,
                                                    const std::vector<TensorScalar<F>> &X) const {
    int err = 0;

    double ev;
    std::unique_ptr<F[]> buff;

    auto idx = Util::split(sub[0], ",");
    std::vector<UInt> vec_dim(idx.size());
    for (unsigned int i = 0; i < vec_dim.size(); i++)
      vec_dim[i] = seq.dim_map.at(idx[i]);

    Tensor<F> T(vec_dim);

    int initSize = 0;
    if (_useInitial) {
      initSize = 1;
      Tensor<F> T0;
      T0(sub[0]) = seq1;
      buff = std::move(T0.data);
    } else {
      buff = std::make_unique<F[]>(T.totalDim);
    }

    err = Algebra::tensorEigen(&ev, buff.get(), sub, seq, P, X,
                               Algebra::Options(1, tolerance, initSize, Algebra::Target::smallest, {}));

    T.data = std::move(buff);

    return std::make_tuple(ev, std::move(T));
  }

  template <typename F>
  EigenSolver<F> &EigenSolver<F>::operator()(const std::string &sub1, const std::string &sub2) {
    sub = {sub1, sub2};
    return *this;
  }

  template <typename F>
  EigenSolver<F> &EigenSolver<F>::operator()(const std::array<std::string, 2> &sub1) {
    sub = sub1;
    return *this;
  }
} // namespace TNT::Tensor

template class TNT::Tensor::EigenSolver<double>;
template class TNT::Tensor::EigenSolver<std::complex<double>>;
