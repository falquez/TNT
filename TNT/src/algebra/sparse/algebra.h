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

#ifndef _TNT_SPARSE_ALGEBRA_H
#define _TNT_SPARSE_ALGEBRA_H

#include "../../util/util.h"
#include "../algebra.h"

namespace TNT::Algebra::Sparse {

  template <typename F>
  struct SparseTensorData {
    const Tensor::Sparse::Tensor<F> *vec;
    std::array<std::vector<Algebra::UInt>, 2> dim;
    std::array<std::string, 2> sub;
    std::string subM;
    const std::vector<TNT::Tensor::Projector<F>> &P;
    const std::vector<TNT::Tensor::Tensor<F>> &X;
  };

  template <typename F>
  struct SparseTensorData2 {
    SparseTensorData2(const Tensor::Sparse::Tensor<F> *vec, const std::array<std::vector<uint>, 3> &dim,
                      const std::array<std::vector<int>, 3> &idx)
        : vec(vec), dim(dim) {
      subscript[0] = Util::stringify(idx[0]);
      subscript[1] = Util::stringify(idx[1]);
      subscript[2] = Util::stringify(idx[2]);
    }
    const Tensor::Sparse::Tensor<F> *vec;
    std::array<std::vector<uint>, 3> dim;
    std::array<std::string, 3> subscript;
  };

  /*
   * QR decomposition of tensor in M in data,
   * M_{0123..} -> M_{mn} -> Q_{mr} R_{rn}
   * where m=idx[0] and n=idx[1], and dim(m) >= dim(n)
   * Result is written in Q, with the correct transposition, ie
   * Q_{idx[0]idx[1]} -> Q_{0123..}
   * R index position remains same
   */
  template <typename F>
  int tensorQRD(const std::vector<UInt> &dim, const std::array<std::vector<UInt>, 2> &idx, F *data, F *R);

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub, const Tensor::Sparse::Tensor<F> *vec,
                  const Options &options = Options{});

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub,
                  const Tensor::Sparse::Contraction<F> &seq, const std::vector<TNT::Tensor::Projector<F>> &P = {},
                  const std::vector<TNT::Tensor::Tensor<F>> &X = {}, const Options &options = Options{});

  template <typename F>
  int tensorSVD(const Tensor::Sparse::Tensor<F> &vec, const std::array<std::vector<UInt>, 2> &idx, double *svals,
                F *svecs, const Options &options);
} // namespace TNT::Algebra::Sparse

#endif //_TNT_SPARSE_ALGEBRA_H
