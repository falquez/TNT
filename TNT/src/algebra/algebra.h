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

#ifndef _TNT_ALGEBRA_H
#define _TNT_ALGEBRA_H

#include <map>
#include <string>
#include <vector>

#include <TNT/tensor/contraction.h>
#include <TNT/tensor/sparse/contraction.h>
#include <TNT/tensor/sparse/tensor.h>

namespace TNT::Algebra {
  using UInt = unsigned int;

  const int numThreads = 1;

  enum class Target { largest, smallest, closest_leq, closest_abs };

  struct Options {
    UInt nv = 1;
    double tolerance = 1e-9;
    double aNorm = -1.0;
    Target target = Target::smallest;
    int initial = 0;
    std::vector<double> targets = {};
    UInt maxIter = 1000;
    UInt verbosity = 0;

    Options() {}
    Options(const UInt &nv) : nv{nv} {}
    Options(const UInt &nv, const double &tolerance, const double &aNorm)
	: nv{nv}, tolerance{tolerance}, aNorm{aNorm} {}

    Options(const UInt &nv, const double &tolerance, const double &aNorm, const Target &target)
	: nv{nv}, tolerance{tolerance}, aNorm{aNorm}, target{target} {}

    Options(const UInt &nv, const double &tolerance, const double &aNorm, const int &initial, const Target &target)
	: nv{nv}, tolerance{tolerance}, aNorm{aNorm}, target{target}, initial{initial} {}

    Options(const UInt &nv, const double &tolerance, const int &initial, const Target &target,
            const std::vector<double> &targets)
	: nv{nv}, tolerance{tolerance}, initial{initial}, target{target}, targets{targets} {}
  };

  template <typename F>
  int transpose(const std::vector<UInt> &dim, const std::vector<UInt> &trs, F *source, F *target, double alpha = 1.0,
		double beta = 0.0);

  template <typename F>
  int transpose(const std::vector<int> &dim, const std::vector<int> &ldA, const std::vector<int> &ldB,
		const std::vector<int> &trs, F *A, F *B);

  template <typename F>
  int tensorMult(const std::array<std::vector<int>, 3> &dims, const std::array<std::string, 3> &subscripts,
		 const std::array<F *, 3> &data, const double &gamma = 0.0);

  template <typename F>
  int tensorMult(F *result, const std::string subscript, const Tensor::Contraction<F> &seq);

  template <typename F>
  int tensorQRD(const std::array<int, 2> &dim, F *data, F *R);

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
  int tensorLQD(const std::vector<int> &dim, const std::array<std::vector<int>, 2> &idx,
                const std::array<std::vector<int>, 2> &trs, F *data, F *L, F *Q);

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub, const Tensor::Contraction<F> &seq,
		  const std::vector<TNT::Tensor::Tensor<F>> &P = {}, const std::vector<TNT::Tensor::Tensor<F>> &X = {},
		  const Options &options = Options{});

  /* @TODO: Document side data structure
   */
  template <typename F>
  int tensorSVD(int idx, const std::vector<UInt> &dim, F *data, F *side, const Options &options = Options{});

  template <typename F>
  int tensorSVD(const std::vector<UInt> &dim, const std::array<std::vector<UInt>, 2> &idx, double *svals, F *svecs,
		F *data, const Options &options);

} // namespace TNT::Algebra

#endif //_TNT_ALGEBRA_H
