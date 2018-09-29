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

#include <algorithm>
#include <memory>
#include <optional>

//#include "../extern/blas.h"
#include "../util/util.h"
#include "algebra.h"

#include <hptt/hptt.h>
#include <tcl/tcl.h>

namespace TNT::Algebra {

  template <typename F>
  int tensorMult(const std::array<std::vector<int>, 3> &dims,
                 const std::array<std::string, 3> &subscripts, const std::array<F *, 3> &data,
                 const double &gamma) {
    int err = 0;

    std::array<tcl::Tensor<F>, 3> T;

    for (int i = 0; i < 3; i++)
      T[i] = tcl::Tensor<F>(dims[i], data[i]);

    err = tcl::tensorMult<F>(1.0, T[0][subscripts[0]], T[1][subscripts[1]], gamma,
                             T[2][subscripts[2]]);

    return err;
  }

  template <typename F>
  int tensorMult(F *result, const std::string subscript, const Tensor::Contraction<F> &seq) {
    int err = 0;
    int prev = 0, curr = 0;

    std::array<std::unique_ptr<F[]>, 2> buff;
    std::array<std::string, 2> sub;
    std::array<std::vector<std::string>, 2> idx;
    std::array<tcl::Tensor<F>, 2> T;
    std::vector<int> dims;

    sub[curr] = seq.subs[0];
    idx[curr] = Util::split(seq.subs[0], ",");
    std::sort(idx[curr].begin(), idx[curr].end());
    dims = std::vector<int>(seq.dims[0].begin(), seq.dims[0].end());
    T[curr] = tcl::Tensor<F>(dims, seq.data[0]);

    for (UInt i = 1; i < seq.data.size(); i++) {
      prev = (i - 1) % 2;
      curr = i % 2;

      sub[curr] = seq.subs[i];
      idx[curr] = Util::split(seq.subs[i], ",");
      std::sort(idx[curr].begin(), idx[curr].end());

      dims = std::vector<int>(seq.dims[i].begin(), seq.dims[i].end());
      T[curr] = tcl::Tensor<F>(dims, seq.data[i]);

      std::vector<std::string> idx3;
      std::set_symmetric_difference(idx[prev].begin(), idx[prev].end(), idx[curr].begin(),
                                    idx[curr].end(), std::back_inserter(idx3));
      std::string sub3 = Util::concat(idx3, ",");

      std::vector<int> dim3(idx3.size());
      for (UInt i = 0; i < idx3.size(); i++)
        dim3[i] = seq.dim_map.at(idx3[i]);
      std::vector<unsigned long long> dim3l(dim3.begin(), dim3.end());
      unsigned long long size3 = Util::multiply<unsigned long long>(dim3l);

      buff[curr] = std::make_unique<F[]>(size3);
      tcl::Tensor<F> T3(dim3, buff[curr].get());

      err = tcl::tensorMult<F>(1.0, T[prev][sub[prev]], T[curr][sub[curr]], 0.0, T3[sub3]);

      if (err != 0) {
        std::cout << "TensorMult Error: " << err << std::endl;
        exit(err);
      }

      T[curr] = T3;
      idx[curr] = idx3;
      sub[curr] = sub3;
    }

    std::vector<std::string> idxA = Util::split(sub[curr], ",");
    std::vector<std::string> idxB = Util::split(subscript, ",");
    std::vector<int> dimA(idxA.size());
    std::vector<int> perm(idxA.size());

    for (int i = 0; i < idxA.size(); i++) {
      dimA[i] = seq.dim_map.at(idxA[i]);
      perm[i] = find(idxA.begin(), idxA.end(), idxB[i]) - idxA.begin();
    }

    // Permute result
    auto plan = hptt::create_plan(&perm[0], dimA.size(), 1.0, buff[curr].get(), &dimA[0], NULL, 0.0,
                                  result, NULL, hptt::ESTIMATE, Algebra::numThreads);
    plan->execute();

    return err;
  }
} // namespace TNT::Algebra

template int TNT::Algebra::tensorMult<double>(const std::array<std::vector<int>, 3> &dims,
                                              const std::array<std::string, 3> &subscripts,
                                              const std::array<double *, 3> &data,
                                              const double &gamma);
template int TNT::Algebra::tensorMult<std::complex<double>>(
    const std::array<std::vector<int>, 3> &dims, const std::array<std::string, 3> &subscripts,
    const std::array<std::complex<double> *, 3> &data, const double &gamma);

template int TNT::Algebra::tensorMult<double>(double *result, const std::string subscript,
                                              const Tensor::Contraction<double> &seq);
template int TNT::Algebra::tensorMult<std::complex<double>>(
    std::complex<double> *result, const std::string subscript,
    const Tensor::Contraction<std::complex<double>> &seq);
