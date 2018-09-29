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

#include <complex>
#include <iostream>

#include <TNT/tensor/sparse/eigensolver.h>
#include <TNT/tensor/sparse/tensor.h>

#include "../../algebra/sparse/algebra.h"
#include "../../util/util.h"

namespace TNT::Tensor::Sparse {

  template <typename F>
  std::tuple<F, Tensor<F>> EigenSolver<F>::optimize(Tensor<F> &t1) const {
    int err = 0;
    Tensor<F> T(t1.dim);

    double ev = 0.0;
    throw std::invalid_argument("Not Implemented");

    // auto evals = std::make_unique<double[]>(nev);

    // int err = Algebra::Sparse::tensorEigen(&ev, buff.get(), sub, seq,
    //                                       Algebra::Options(1, tolerance, 0,
    //                                       Algebra::Target::closest_leq,
    //                                       targets) );

    // return std::make_tuple(ev,std::move(T));
  }

  template <typename F>
  std::tuple<F, Tensor<F>> EigenSolver<F>::optimize(const Contraction<F> &seq1) const {

    auto idx = Util::split(sub[0], ",");
    std::vector<UInt> dim(idx.size());
    for (unsigned int i = 0; i < dim.size(); i++)
      dim[i] = seq1.dim_map.at(idx[i]);

    Tensor<F> T(dim);
    double ev;

    std::unique_ptr<F[]> buff = std::make_unique<F[]>(T.totalDim);

    int err = Algebra::Sparse::tensorEigen(&ev, buff.get(), sub, seq, {}, {},
					   Algebra::Options(1, tolerance, 0, Algebra::Target::closest_leq, targets));

    T.readFrom(buff.get());

    return std::make_tuple(ev, std::move(T));
  }

  /*template <typename F>
  std::tuple<F, TNT::Tensor::Tensor<F>> EigenSolver<F>::optimize(const TNT::Tensor::Contraction<F> &seq1) const {
    int err = 0;

    double ev;
    std::unique_ptr<F[]> buff;

    auto idx = Util::split(sub[0], ",");
    std::vector<UInt> vec_dim(idx.size());
    for (unsigned int i = 0; i < vec_dim.size(); i++)
      vec_dim[i] = seq.dim_map.at(idx[i]);

    TNT::Tensor::Tensor<F> T(vec_dim);

    int initSize = 0;
    if (_useInitial) {
      initSize = 1;
      TNT::Tensor::Tensor<F> T0;
      T0(sub[0]) = seq1;
      buff = std::move(T0.data); // std::make_unique<F[]>(T.totalDim);
    } else {
      buff = std::make_unique<F[]>(T.totalDim);
    }

    err = Algebra::Sparse::tensorEigen(&ev, buff.get(), sub, seq, {},
				       Algebra::Options(1, tolerance, initSize, Algebra::Target::smallest, {}));

    T.data = std::move(buff);

    return std::make_tuple(ev, std::move(T));
  }*/

  template <typename F>
  std::tuple<F, TNT::Tensor::Tensor<F>> EigenSolver<F>::optimize(const TNT::Tensor::Tensor<F> &t1) const {
    int err = 0;

    double ev;
    std::unique_ptr<F[]> buff;

    TNT::Tensor::Tensor<F> T(t1.dim);

    int initSize = 0;
    buff = std::make_unique<F[]>(T.totalDim);

    if (_useInitial) {
      initSize = 1;
      t1.writeTo(buff.get());
    }

    err = Algebra::Sparse::tensorEigen(&ev, buff.get(), sub, seq, {}, {},
				       Algebra::Options(1, tolerance, initSize, Algebra::Target::smallest, {}));

    T.data = std::move(buff);

    return std::make_tuple(ev, std::move(T));
  }

  template <typename F>
  std::tuple<F, TNT::Tensor::Tensor<F>> EigenSolver<F>::optimize(const TNT::Tensor::Contraction<F> &seq1,
								 const std::vector<TNT::Tensor::Tensor<F>> &P,
								 const std::vector<TNT::Tensor::Tensor<F>> &X) const {
    int err = 0;

    double ev;
    std::unique_ptr<F[]> buff;

    auto idx = Util::split(sub[0], ",");
    std::vector<UInt> vec_dim(idx.size());
    for (unsigned int i = 0; i < vec_dim.size(); i++)
      vec_dim[i] = seq.dim_map.at(idx[i]);

    TNT::Tensor::Tensor<F> T(vec_dim);

    int initSize = 0;
    if (_useInitial) {
      initSize = 1;
      TNT::Tensor::Tensor<F> T0;
      T0(sub[0]) = seq1;
      buff = std::move(T0.data); // std::make_unique<F[]>(T.totalDim);
    } else {
      buff = std::make_unique<F[]>(T.totalDim);
    }

    err = Algebra::Sparse::tensorEigen(&ev, buff.get(), sub, seq, P, X,
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

} // namespace TNT::Tensor::Sparse

template class TNT::Tensor::Sparse::EigenSolver<double>;
template class TNT::Tensor::Sparse::EigenSolver<std::complex<double>>;
