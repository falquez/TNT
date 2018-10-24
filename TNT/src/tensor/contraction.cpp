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
#include <cassert>
#include <complex>
#include <iostream>

#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

#include "../algebra/algebra.h"
#include "../extern/blas.h"
#include "../util/util.h"

namespace TNT::Tensor {

  template <typename F>
  Contraction<F>::Contraction(const Tensor<F> &t) : dims{t.dim}, subs{t.sub}, data{t.data.get()}, dim_map{} {

    std::vector<std::string> idx = Util::split(subs[0], ",");

    for (UInt i = 0; i < dims[0].size(); i++)
      dim_map.emplace(idx[i], dims[0][i]);
  }

  template <typename F>
  Contraction<F> &Contraction<F>::operator*(const Tensor<F> &t) {

    data.push_back(t.data.get());
    subs.push_back(t.sub);
    dims.push_back(t.dim);
    // strides.push_back(t.stride);
    // totalDim.push_back(t.totalDim);

    std::vector<std::string> idx = Util::split(subs.back(), ",");
    for (UInt i = 0; i < dims.back().size(); i++)
      dim_map.emplace(idx[i], dims.back()[i]);

    return *this;
  }

  template <typename F>
  Contraction<F>::operator F() const {
    assert(data.size() == 2);

    auto totalDim = Util::multiply(dims[0]);

    return TNT::BLAS::dot<F>(totalDim, {data[0], data[1]});
  }

  template <typename F>
  std::tuple<Tensor<F>, Tensor<F>> Contraction<F>::SVD(std::array<std::string, 2> subscript,
                                                       const SVDOptions &svdopts) const {
    int err = 0;

    std::array<Tensor<F>, 2> svd;
    std::array<std::vector<UInt>, 2> dim_tr;
    std::array<std::vector<UInt>, 2> links_tr;
    std::array<std::vector<std::string>, 2> subs;
    std::array<std::string, 2> svd_subs;
    std::array<UInt, 2> svd_dim;

    std::vector<std::string> svd_sub;
    std::map<std::string, UInt> svd_map = dim_map;

    // Read and sort result tensor subscripts
    for (UInt n = 0; n < 2; n++) {
      subs[n] = Util::split(subscript[n], ",");
      std::sort(subs[n].begin(), subs[n].end());
    }
    // Find SVD subscript as the intersection
    std::set_intersection(subs[0].begin(), subs[0].end(), subs[1].begin(), subs[1].end(), std::back_inserter(svd_sub));
    // Reread tensor subscripts
    for (UInt n = 0; n < 2; n++)
      subs[n] = Util::split(subscript[n], ",");

    // Fill svd_subs
    for (UInt n = 0; n < 2; n++) {
      std::vector<std::string> idx;
      svd_dim[n] = 1;
      for (const auto &s : subs[n]) {
        if (s != svd_sub[0]) {
          idx.push_back(s);
          svd_dim[n] = svd_dim[n] * dim_map.at(s);
        }
      }
      svd_subs[n] = Util::concat(idx);
    }

    UInt dimM = svd_dim[0];
    UInt dimN = svd_dim[1];

    // std::cout << "Contraction<F>::SVD dimM=" << dimM << " dimN=" << dimN << " nsv=" << svdopts.nsv << std::endl;
    std::unique_ptr<double[]> svals = std::make_unique<double[]>(svdopts.nsv);
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>((dimM + dimN) * svdopts.nsv);

    // std::unique_ptr<double[]> svals = std::make_unique<double[]>(dimM);
    // std::unique_ptr<F[]> svecs = std::make_unique<F[]>(dimM * dimM + dimN * dimN);

    // double anorm = norm2();
    auto opts = Algebra::Options(svdopts.nsv, svdopts.tolerance, Algebra::Target::largest);
    UInt nvecs = Algebra::tensorSVD<F>(svd_subs, *this, svals.get(), svecs.get(), opts);

    svd_map[svd_sub[0]] = nvecs;

    std::unique_ptr<F[]> lvecs = std::make_unique<F[]>(dimM * nvecs);
    std::unique_ptr<F[]> rvecs = std::make_unique<F[]>(dimN * nvecs);

    // UInt ldM = nvecs;
    switch (svdopts.norm) {
    case SVDNorm::equal:
      for (UInt n = 0; n < nvecs; n++) {
        for (UInt i = 0; i < dimM; i++)
          lvecs[n * dimM + i] = svecs[n * dimM + i] * std::sqrt(svals[n]);
        for (UInt i = 0; i < dimN; i++)
          rvecs[n * dimN + i] = svecs[nvecs * dimM + n * dimN + i] * std::sqrt(svals[n]);
      }
      break;
    case SVDNorm::left:
      for (UInt n = 0; n < nvecs; n++) {
        for (UInt i = 0; i < dimM; i++)
          lvecs[n * dimM + i] = svecs[n * dimM + i];
        for (UInt i = 0; i < dimN; i++)
          rvecs[n * dimN + i] = svecs[nvecs * dimM + n * dimN + i] * svals[n];
      }
      break;
    case SVDNorm::right:
      for (UInt n = 0; n < nvecs; n++) {
        for (UInt i = 0; i < dimM; i++)
          lvecs[n * dimM + i] = svecs[n * dimM + i] * svals[n];
        for (UInt i = 0; i < dimN; i++)
          rvecs[n * dimN + i] = svecs[nvecs * dimM + n * dimN + i];
      }
      break;
    }

    /*std::cout << "contraction lvecs={";
    for (UInt n = 0; n < dimM * nvecs; n++) {
      std::cout << lvecs[n] << ",";
    }
    std::cout << "}" << std::endl;
    std::cout << "contraction rvecs={";
    for (UInt n = 0; n < dimN * nvecs; n++) {
      std::cout << rvecs[n] << ",";
    }
    std::cout << "}" << std::endl;*/
    // Initialize dimension map
    // for (UInt i = 0; i < index.size(); i++)
    //  dim_map[index[i]] = dim[i];
    // Only single subscript used

    // Initialize svd[n] with correct dimension
    for (UInt n = 0; n < 2; n++) {
      std::vector<UInt> svd_dim;
      for (const auto &d : subs[n])
        svd_dim.push_back(svd_map[d]);
      svd[n] = Tensor<F>(svd_dim);
    }

    // Determine correct dimensions of lvecs and rvecs
    for (UInt n = 0; n < 2; n++) {
      for (const auto &s : subs[n]) {
        if (s != svd_sub[0])
          dim_tr[n].push_back(svd_map.at(s));
      }
      dim_tr[n].push_back(nvecs);
    }
    // Determine correct transposition
    // Since links[n] are alredy "almost correctly transposed"
    // Only insert svd index at correct position
    for (UInt n = 0; n < 2; n++) {
      for (UInt i = 0; i < subs[n].size() - 1; i++)
        links_tr[n].push_back(i);
      // Find position of svd subscript in subs[n]
      auto iter = std::find(subs[n].begin(), subs[n].end(), svd_sub[0]);
      // insert idx=links[n].size() at position
      links_tr[n].insert(links_tr[n].begin() + std::distance(subs[n].begin(), iter), subs[n].size() - 1);
    }

    /*std::cout << "dim_tr[0]=(";
    for (const auto &d : dim_tr[0])
      std::cout << d << ",";
    std::cout << "), dim_tr[1]=(";
    for (const auto &d : dim_tr[1])
      std::cout << d << ",";
    std::cout << "), links_tr[0]=(";
    for (const auto &d : links_tr[0])
      std::cout << d << ",";
    std::cout << "), links_tr[1]=(";
    for (const auto &d : links_tr[1])
      std::cout << d << ",";
    std::cout << ")" << std::endl;*/
    err = Algebra::transpose(dim_tr[0], links_tr[0], lvecs.get(), svd[0].data.get());
    err = Algebra::transpose(dim_tr[1], links_tr[1], rvecs.get(), svd[1].data.get());

    return std::make_tuple(std::move(svd[0]), std::move(svd[1]));
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Contraction<F> &T) {
    out << "Contraction: (Not Implemented)\n";
    return out;
  }
} // namespace TNT::Tensor

template class TNT::Tensor::Contraction<double>;
template class TNT::Tensor::Contraction<std::complex<double>>;

template std::ostream &TNT::Tensor::operator<<<double>(std::ostream &out, const Contraction<double> &T);
template std::ostream &TNT::Tensor::operator<<<std::complex<double>>(std::ostream &out,
                                                                     const Contraction<std::complex<double>> &T);
