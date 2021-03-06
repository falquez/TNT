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
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <random>

#include <TNT/storage/storage.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

#include "../algebra/algebra.h"
#include "../extern/blas.h"
#include "../util/util.h"

namespace TNT::Tensor {

  std::vector<UInt> convertIndexL(const std::vector<UInt> &stride, const ULong &i) {
    std::vector<UInt> idx(stride.size());
    idx.back() = i / stride.back();
    for (ulong p = 0; p < idx.size() - 1; p++)
      idx[p] = (i % stride[p + 1]) / stride[p];
    return idx;
  }

  ULong convertIndexL(const std::vector<UInt> &stride, const std::vector<UInt> &idx) {
    ULong i = 0;
    for (ulong p = 0; p < stride.size(); p++)
      i += idx[p] * stride[p];
    return i;
  }

  template <typename F>
  Tensor<F>::Tensor(const std::vector<UInt> &dim) : dim{dim}, stride(dim.size()), sub{} {

    totalDim = Util::multiply(dim);
    stride[0] = 1;
    for (UInt i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    data = std::make_unique<F[]>(totalDim);
  }

  template <typename F>
  Tensor<F>::Tensor(const std::string &filename, const std::string &path, const unsigned int &id) {
    Storage::Storage storage(filename, Storage::FileMode::ReadOnly);
    Storage::Data::Dense<F> dense;

    storage.read(path + "/" + std::to_string(id), dense);

    std::vector<UInt> uidim(dense.dim.begin(), dense.dim.end());
    dim = uidim;
    stride.resize(dim.size());
    totalDim = Util::multiply(dim);
    stride[0] = 1;
    for (UInt i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    data = std::move(dense.data);
  }

  template <typename F>
  int Tensor<F>::writeToFile(const std::string &filename, const std::string &path, const unsigned int &id) const {
    Storage::Storage storage(filename, Storage::FileMode::CreateOverwrite);
    Storage::Data::Dense<F> dense;

    dense.data = std::make_unique<F[]>(totalDim);
    for (const auto &d : dim)
      dense.dim.push_back(d);
    writeTo(dense.data.get());

    storage.create_group(path);
    storage.create(path + "/" + std::to_string(id), dense);

    return 0;
  }

  template <typename F>
  Tensor<F>::Tensor(const std::vector<UInt> &dim, const F initial) : dim{dim}, stride(dim.size()), sub{} {

    totalDim = Util::multiply(dim);
    stride[0] = 1;
    for (UInt i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    data = std::make_unique<F[]>(totalDim);
    for (UInt i = 0; i < totalDim; i++)
      data[i] = initial;
  }

  template <typename F>
  Tensor<F>::Tensor(const Tensor<F> &t) {
    totalDim = t.totalDim;

    dim = t.dim;
    stride = t.stride;
    sub = t.sub;

    data = std::make_unique<F[]>(totalDim);
    for (UInt i = 0; i < totalDim; i++)
      data[i] = t.data[i];
  }

  template <typename F>
  Tensor<F>::Tensor(std::tuple<std::vector<UInt>, std::unique_ptr<F[]>> &&d) : dim{std::get<0>(d)}, stride(dim.size()) {
    // dim = std::get<0>(d);
    totalDim = Util::multiply(dim);
    stride[0] = 1;
    for (UInt i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];
    data = std::move(std::get<1>(d));
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator=(const Tensor<F> &t) {

    totalDim = t.totalDim;

    dim = t.dim;
    stride = t.stride;
    sub = t.sub;

    data = std::make_unique<F[]>(totalDim);
    for (UInt i = 0; i < totalDim; i++)
      data[i] = t.data[i];

    return *this;
  }

  template <typename F>
  F Tensor<F>::dot(const F *p) const {
    return TNT::BLAS::dot<F>(totalDim, {data.get(), p});
  }

  template <typename F>
  Tensor<F> &Tensor<F>::merge(const std::string &mrg) {
    std::map<std::string, UInt> idx_map;

    std::vector<std::string> mrg_idx = Util::split(mrg, ",");
    std::vector<std::string> sub_idx = Util::split(sub, ",");
    for (UInt i = 0; i < sub_idx.size(); i++)
      idx_map.emplace(sub_idx[i], i);

    assert(mrg_idx.size() == 2 && ((idx_map.at(mrg_idx[1]) - idx_map.at(mrg_idx[0])) == 1));

    std::vector<UInt> ndim(dim.size() - 1);
    std::vector<UInt> nstride(dim.size() - 1);
    UInt m = idx_map.at(mrg_idx[0]);
    for (UInt i = 0; i < m; i++)
      ndim[i] = dim[i];
    ndim[m] = dim[m] * dim[m + 1];
    for (UInt i = m + 1; i < dim.size(); i++)
      ndim[i] = dim[i + 1];

    nstride[0] = 1;
    for (UInt i = 1; i < ndim.size(); i++)
      nstride[i] = ndim[i - 1] * nstride[i - 1];

    dim = ndim;
    stride = nstride;

    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::split(const std::string &spl) {
    throw std::invalid_argument("Not Implemented");

    return *this;
  }

  template <typename F>
  Tensor<F> Tensor<F>::transpose(const std::vector<UInt> tidx) const {
    assert(tidx.size() == dim.size());

    std::vector<UInt> tdim(dim.size());
    for (uint i = 0; i < dim.size(); i++)
      tdim[i] = dim[tidx[i]];
    Tensor<F> trans(tdim);

    for (UInt l = 0; l < totalDim; l++) {
      std::vector<UInt> idx_orig = convertIndexL(stride, l);
      std::vector<UInt> idx(dim.size());
      for (UInt i = 0; i < dim.size(); i++)
        idx[i] = idx_orig[tidx[i]];
      trans[idx] = data[l];
    }

    return trans;
  }

  template <typename F>
  Tensor<F> Tensor<F>::matricize(const std::vector<UInt> &idx_r, const std::vector<UInt> &idx_c) const {
    assert(idx_r.size() + idx_c.size() == dim.size());

    UInt dim_r, dim_c;
    std::vector<UInt> stride_r(idx_r.size());
    std::vector<UInt> stride_c(idx_c.size());

    dim_r = dim[idx_r[0]];
    stride_r[0] = 1;
    for (uint i = 1; i < idx_r.size(); i++) {
      dim_r *= dim[idx_r[i]];
      stride_r[i] = stride_r[i - 1] * dim[idx_r[i]];
    }
    dim_c = dim[idx_c[0]];
    stride_c[0] = 1;
    for (uint i = 1; i < idx_c.size(); i++) {
      dim_c *= dim[idx_c[i]];
      stride_c[i] = stride_c[i - 1] * dim[idx_c[i]];
    }
    Tensor<F> mat({dim_r, dim_c});

    /*for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin(); it != data.end(); it++) {
      std::vector<UInt> idx{0, 0};
      for (uint i = 0; i < idx_r.size(); i++)
        idx[0] += it->first[idx_r[i]] * stride_r[i];
      for (uint i = 0; i < idx_c.size(); i++)
        idx[1] += it->first[idx_c[i]] * stride_c[i];
      mat.data.insert({idx, it->second});
    }*/

    for (UInt l = 0; l < totalDim; l++) {
      std::vector<UInt> idx_orig = convertIndexL(stride, l);
      std::vector<UInt> idx{0, 0};
      // for (UInt j = 0; j < dim.size(); j++)
      // idx[j] = idx_orig[tidx[j]];
      for (uint i = 0; i < idx_r.size(); i++)
        idx[0] += idx_orig[idx_r[i]] * stride_r[i];
      for (uint i = 0; i < idx_c.size(); i++)
        idx[1] += idx_orig[idx_c[i]] * stride_c[i];
      mat[idx] = data[l];
    }

    return mat;
  }

  template <typename F>
  Tensor<F> Tensor<F>::trace(const std::string &tr) const {

    std::unique_ptr<F[]> tr_data = std::make_unique<F[]>(totalDim);
    std::vector<std::string> idx_s = Util::split(sub, ",");
    std::vector<std::string> tr_sub = Util::split(tr, ",");
    std::map<std::string, UInt> idx_map;
    for (UInt i = 0; i < idx_s.size(); i++)
      idx_map.emplace(idx_s[i], i);
    std::array<UInt, 2> tr_idx = {idx_map.at(tr_sub[0]), idx_map.at(tr_sub[1])};

    // assert(tr_sub.size() == 2);

    assert(idx_s.size() == dim.size() && dim[tr_idx[0]] == dim[tr_idx[1]]);
    std::vector<UInt> trsp, ndim;
    trsp.push_back(tr_idx[0]);
    trsp.push_back(tr_idx[1]);
    for (UInt i = 0; i < dim.size(); i++)
      if (i != tr_idx[0] && i != tr_idx[1]) {
        trsp.push_back(i);
        ndim.push_back(dim[i]);
      }

    int err = Algebra::transpose(dim, trsp, data.get(), tr_data.get());

    Tensor<F> T(ndim);

    // Sum over trace and copy to T
    UInt tr_dim = dim[tr_idx[0]];
    for (UInt i = 0; i < T.totalDim; i++) {
      for (UInt j = 0; j < tr_dim; j++) {
        T.data[i] += tr_data[i * tr_dim * tr_dim + j * tr_dim + j];
      }
    }

    return std::move(T);
  }

  /*template <typename F>
  Tensor<F>::Tensor(const Sparse::Tensor<F> &t) : dim{t.dimension()}, stride(dim.size()), sub{} {

    totalDim = Util::multiply(dim);

    stride[0] = 1;
    for (UInt i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    data = std::make_unique<F[]>(totalDim);

    for (const auto [idx, v] : t.elements()) {
      ULong j = 0;
      for (int p = 0; p < stride.size(); p++)
        j += idx[p] * stride[p];
      data[j] = v;
    }
  }*/

  template <typename F>
  double Tensor<F>::norm2() const {
    double nrm = 0;
    for (UInt i = 0; i < totalDim; i++)
      nrm += std::norm(data[i]);
    return std::sqrt(nrm);
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator+=(const Tensor<F> &M) {
    // Tensor<F> T(dim);
    for (UInt i = 0; i < totalDim; i++) {
      data[i] += M.data[i];
    }
    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator*=(const Tensor<F> &M) {
    return *this;
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator+(const Tensor<F> &M) const {
    Tensor<F> T(dim);
    for (UInt i = 0; i < totalDim; i++) {
      T.data[i] = data[i] + M.data[i];
    }
    return std::move(T);
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator-(const Tensor<F> &M) const {
    Tensor<F> T(dim);
    for (UInt i = 0; i < totalDim; i++) {
      T.data[i] = data[i] - M.data[i];
    }
    return std::move(T);
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator-() const {
    Tensor<F> T(dim);
    for (UInt i = 0; i < totalDim; i++) {
      T.data[i] = -data[i];
    }
    return std::move(T);
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator*(const F &c) const {
    Tensor<F> M(dim);
    for (UInt i = 0; i < totalDim; i++) {
      M[i] = data[i] * c;
    }
    return std::move(M);
  }

  template <typename F>
  Contraction<F> Tensor<F>::operator*(const Tensor<F> &rhs) const {
    return Contraction<F>(*this) * rhs;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator=(const Contraction<F> &tc) {
    int err = 0;

    const UInt nc = tc.data.size();

    auto idx = Util::split(sub, ",");
    dim = std::vector<UInt>(idx.size());
    stride = std::vector<UInt>(idx.size());

    for (UInt i = 0; i < dim.size(); i++)
      dim[i] = tc.dim_map.at(idx[i]);

    stride[0] = 1;
    for (UInt i = 1; i < stride.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    totalDim = Util::multiply(dim);

    // Allocate new memory for result
    std::unique_ptr<F[]> result = std::make_unique<F[]>(totalDim);

    err = Algebra::tensorMult<F>(result.get(), sub, tc);

    // move result to data
    data = std::move(result);

    // clear subscripts
    sub.clear();
    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::initialize(const int &mod) {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (UInt i = 0; i < totalDim; i++)
      data[i] = d(rng);
    return *this;
  }

  template <typename F>
  Tensor<F> Tensor<F>::conjugate() const {
    Tensor<F> conj(dim);
    for (UInt i = 0; i < totalDim; i++)
      conj.data[i] = data[i];
    return conj;
  }

  template <>
  Tensor<std::complex<double>> Tensor<std::complex<double>>::conjugate() const {
    Tensor<std::complex<double>> conj(dim);
    for (UInt i = 0; i < totalDim; i++)
      conj.data[i] = std::conj(data[i]);
    return conj;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::readFrom(const F *source, const double &eps) {
    for (ULong i = 0; i < totalDim; i++)
      data[i] = source[i];

    return *this;
  }

  template <typename F>
  const Tensor<F> &Tensor<F>::writeTo(F *target, const double &eps) const {
    for (ULong i = 0; i < totalDim; i++)
      target[i] = data[i];

    return *this;
  }

  template <typename F>
  const Tensor<F> &Tensor<F>::addTo(F *target, const F &alpha, const double &eps) const {
    for (ULong i = 0; i < totalDim; i++)
      target[i] += alpha * data[i];

    return *this;
  }

  template <typename F>
  UInt Tensor<F>::dimension(const std::string &s) const {
    unsigned int d = 0;
    auto index = Util::split(sub, ",");
    auto iter = std::find(index.begin(), index.end(), s);
    if (iter != index.end()) {
      d = std::distance(index.begin(), iter);
    } else {
      throw std::out_of_range("Index not found");
    }
    return dim[d];
  }

  template <typename F>
  Tensor<F> Tensor<F>::normalize_QRD(const UInt &idx_r) {
    Tensor<F> R({dim[idx_r], dim[idx_r]});

    std::array<std::vector<UInt>, 2> links;
    for (UInt i = 0; i < dim.size(); i++)
      if (i != idx_r)
        links[0].push_back(i);
    links[1] = {idx_r};

    int res = Algebra::tensorQRD(dim, links, data.get(), R.data.get());

    return R;
  }

  template <typename F>
  std::tuple<Tensor<F>, Tensor<F>> Tensor<F>::normalize_QRD(const std::string &idx_s) {
    std::vector<std::string> index = Util::split(sub, ",");
    auto iter = std::find(index.begin(), index.end(), idx_s);
    UInt idx_r = std::distance(index.begin(), iter);

    std::array<std::vector<UInt>, 2> links;
    for (UInt i = 0; i < dim.size(); i++)
      if (i != idx_r)
        links[0].push_back(i);
    links[1] = {idx_r};

    Tensor<F> R({dim[idx_r], dim[idx_r]});
    Tensor<F> T(*this);

    int res = Algebra::tensorQRD(dim, links, T.data.get(), R.data.get());

    return std::make_tuple(std::move(T), std::move(R));
  }

  template <typename F>
  std::tuple<Tensor<F>, std::vector<double>, Tensor<F>> Tensor<F>::SVD(std::array<std::string, 2> subscript,
                                                                       const SVDOptions &svdopts) const {
    int err = 0;

    std::array<Tensor<F>, 2> svd;
    std::array<std::vector<UInt>, 2> dim_tr;
    std::array<std::vector<UInt>, 2> links_tr;
    std::array<std::vector<std::string>, 2> subs;
    std::vector<std::string> svd_sub;
    std::array<std::vector<UInt>, 2> links;
    std::map<std::string, UInt> dim_map;

    // This Tensor subscript
    std::vector<std::string> index = Util::split(sub, ",");

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

    // Find the position of free indices in the tensor indices string
    // write to links[n]
    for (UInt n = 0; n < 2; n++) {
      for (const auto &s : subs[n]) {
        auto iter = std::find(index.begin(), index.end(), s);
        if (iter != index.end())
          links[n].push_back(std::distance(index.begin(), iter));
      }
    }

    UInt dimM = Util::multiply(Util::selectU(dim, links[0]));
    UInt dimN = Util::multiply(Util::selectU(dim, links[1]));

    // std::unique_ptr<double[]> svals = std::make_unique<double[]>(svdopts.nsv);
    // std::unique_ptr<F[]> svecs = std::make_unique<F[]>((dimM + dimN) * svdopts.nsv);

    std::unique_ptr<double[]> svals = std::make_unique<double[]>(dimM);
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>(dimM * dimM + dimN * dimN);

    double anorm = norm2();
    auto opts = Algebra::Options(svdopts.nsv, svdopts.tolerance, anorm, Algebra::Target::largest);

    UInt nvecs = Algebra::tensorSVD<F>(dim, links, svals.get(), svecs.get(), data.get(), opts);
    std::unique_ptr<F[]> lvecs = std::make_unique<F[]>(dimM * nvecs);
    std::unique_ptr<F[]> rvecs = std::make_unique<F[]>(dimN * nvecs);

    std::vector<double> svd_values(svals.get(), svals.get() + nvecs);

    UInt ldM = dimM; // nvecs
    switch (svdopts.norm) {
    case SVDNorm::equal:
      for (UInt n = 0; n < nvecs; n++) {
        for (UInt i = 0; i < dimM; i++)
          lvecs[n * dimM + i] = svecs[n * dimM + i] * std::sqrt(svals[n]);
        for (UInt i = 0; i < dimN; i++)
          rvecs[n * dimN + i] = svecs[ldM * dimM + i * dimN + n] * std::sqrt(svals[n]);
      }
      break;
    case SVDNorm::left:
      for (UInt n = 0; n < nvecs; n++) {
        for (UInt i = 0; i < dimM; i++)
          lvecs[n * dimM + i] = svecs[n * dimM + i];
        for (UInt i = 0; i < dimN; i++)
          rvecs[n * dimN + i] = svecs[ldM * dimM + i * dimN + n] * svals[n];
      }
      break;
    case SVDNorm::right:
      for (UInt n = 0; n < nvecs; n++) {
        for (UInt i = 0; i < dimM; i++)
          lvecs[n * dimM + i] = svecs[n * dimM + i] * svals[n];
        for (UInt i = 0; i < dimN; i++)
          rvecs[n * dimN + i] = svecs[ldM * dimM + i * dimN + n];
      }
      break;
    }

    // Initialize dimension map
    for (UInt i = 0; i < index.size(); i++)
      dim_map[index[i]] = dim[i];
    // Only single subscript used
    dim_map[svd_sub[0]] = nvecs;

    // Initialize svd[n] with correct dimension
    for (UInt n = 0; n < 2; n++) {
      std::vector<UInt> svd_dim;
      for (const auto &d : subs[n])
        svd_dim.push_back(dim_map[d]);
      svd[n] = Tensor<F>(svd_dim);
    }

    // Determine correct dimensions of lvecs and rvecs
    for (UInt n = 0; n < 2; n++) {
      for (const auto &d : links[n])
        dim_tr[n].push_back(dim[d]);
      dim_tr[n].push_back(nvecs);
    }
    // Determine correct transposition
    // Since links[n] are alredy "almost correctly transposed"
    // Only insert svd index at correct position
    for (UInt n = 0; n < 2; n++) {
      for (UInt i = 0; i < links[n].size(); i++)
        links_tr[n].push_back(i);
      // Find position of svd subscript in subs[n]
      auto iter = std::find(subs[n].begin(), subs[n].end(), svd_sub[0]);
      // insert idx=links[n].size() at position
      links_tr[n].insert(links_tr[n].begin() + std::distance(subs[n].begin(), iter), links[n].size());
    }

    err = Algebra::transpose(dim_tr[0], links_tr[0], lvecs.get(), svd[0].data.get());
    err = Algebra::transpose(dim_tr[1], links_tr[1], rvecs.get(), svd[1].data.get());

    return std::make_tuple(std::move(svd[0]), std::move(svd_values), std::move(svd[1]));
  }

  template <typename F>
  std::tuple<Tensor<F>, Tensor<F>> Tensor<F>::SVD(std::array<std::string, 2> subscript, const Tensor<F> &left,
                                                  const Tensor<F> &right, const SVDOptions &svdopts) const {

    int err = 0;

    // This Tensor subscript
    std::vector<std::string> index = Util::split(sub, ",");

    // Read and sort result tensor subscripts
    std::array<std::vector<std::string>, 2> subs;
    for (UInt n = 0; n < 2; n++)
      subs[n] = Util::split(subscript[n], ",");
    for (UInt n = 0; n < 2; n++)
      std::sort(subs[n].begin(), subs[n].end());
    // Find SVD subscript as the intersection
    std::vector<std::string> svd_sub;
    std::set_intersection(subs[0].begin(), subs[0].end(), subs[1].begin(), subs[1].end(), std::back_inserter(svd_sub));
    // Reread tensor subscripts
    for (UInt n = 0; n < 2; n++)
      subs[n] = Util::split(subscript[n], ",");

    // Find the position of free indices in the tensor indices string, write to links[n]
    std::array<std::vector<UInt>, 2> links;
    for (UInt n = 0; n < 2; n++) {
      for (const auto &s : subs[n]) {
        auto iter = std::find(index.begin(), index.end(), s);
        if (iter != index.end())
          links[n].push_back(std::distance(index.begin(), iter));
      }
    }

    UInt initSize = svdopts.nsv;
    UInt dimM = Util::multiply(Util::selectU(dim, links[0]));
    UInt dimN = Util::multiply(Util::selectU(dim, links[1]));

    std::unique_ptr<double[]> svals = std::make_unique<double[]>(svdopts.nsv);
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>((dimM + dimN) * svdopts.nsv);
    std::unique_ptr<F[]> lvecs = std::make_unique<F[]>(dimM * svdopts.nsv);
    std::unique_ptr<F[]> rvecs = std::make_unique<F[]>(dimN * svdopts.nsv);

    // Fill svecs with initial left and right vectors.

    // Determine correct transposition
    std::array<std::vector<UInt>, 2> links_tr;
    for (UInt n = 0; n < 2; n++) {
      // Find position of svd subscript in subs[n]
      auto iter = std::find(subs[n].begin(), subs[n].end(), svd_sub[0]);
      UInt idx_pos = std::distance(subs[n].begin(), iter);
      for (UInt i = 0; i < links[n].size(); i++) {
        if (i != idx_pos)
          links_tr[n].push_back(i);
      }
      links_tr[n].push_back(idx_pos);
    }

    err = Algebra::transpose(left.dimension(), links_tr[0], left.data.get(), svecs.get());
    err = Algebra::transpose(right.dimension(), links_tr[1], right.data.get(), svecs.get() + initSize * dimM);

    // Normalize initial left and right vectors
    for (UInt n = 0; n < initSize; n++) {
      double norm;

      norm = 0;
      F *vl = svecs.get() + n * dimM;
      for (UInt i = 0; i < dimM; i++)
        norm += std::norm(vl[i]);
      norm = std::sqrt(norm);
      for (UInt i = 0; i < dimM; i++)
        vl[i] = vl[i] / norm;

      norm = 0;
      F *vr = svecs.get() + initSize * dimM + n * dimN;
      for (UInt i = 0; i < dimN; i++)
        norm += std::norm(vr[i]);
      norm = std::sqrt(norm);
      for (UInt i = 0; i < dimN; i++)
        vr[i] = vr[i] / norm;
    }

    double anorm = norm2();
    auto opts = Algebra::Options(svdopts.nsv, svdopts.tolerance, anorm, initSize, Algebra::Target::largest);
    opts.verbosity = 3;
    UInt nvecs = Algebra::tensorSVD<F>(dim, links, svals.get(), svecs.get(), data.get(), opts);

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

    // Initialize dimension map
    std::map<std::string, UInt> dim_map;
    for (UInt i = 0; i < index.size(); i++)
      dim_map[index[i]] = dim[i];
    // Only single subscript used
    dim_map[svd_sub[0]] = nvecs;

    std::array<Tensor<F>, 2> svd;
    // Initialize svd[n] with correct dimension
    for (UInt n = 0; n < 2; n++) {
      std::vector<UInt> svd_dim;
      for (const auto &d : subs[n])
        svd_dim.push_back(dim_map[d]);
      svd[n] = Tensor<F>(svd_dim);
    }

    // Determine correct dimensions of lvecs and rvecs
    std::array<std::vector<UInt>, 2> dim_tr;
    for (UInt n = 0; n < 2; n++) {
      for (const auto &d : links[n])
        dim_tr[n].push_back(dim[d]);
      dim_tr[n].push_back(nvecs);
    }
    // Determine correct transposition
    // Since links[n] are alredy "almost correctly" transposed
    // Only insert svd index at correct position
    for (UInt n = 0; n < 2; n++) {
      links_tr[n].clear();
      for (UInt i = 0; i < links[n].size(); i++)
        links_tr[n].push_back(i);
      // Find position of svd subscript in subs[n]
      auto iter = std::find(subs[n].begin(), subs[n].end(), svd_sub[0]);
      // insert idx=links[n].size() at position
      links_tr[n].insert(links_tr[n].begin() + std::distance(subs[n].begin(), iter), links[n].size());
    }

    err = Algebra::transpose(dim_tr[0], links_tr[0], lvecs.get(), svd[0].data.get());
    err = Algebra::transpose(dim_tr[1], links_tr[1], rvecs.get(), svd[1].data.get());

    return std::make_tuple(std::move(svd[0]), std::move(svd[1]));
  }

  template <typename F>
  std::tuple<Tensor<F>, Tensor<F>> Tensor<F>::SVD(std::array<std::vector<UInt>, 2> links,
                                                  const SVDOptions &options) const {
    int err = 0;
    std::array<Tensor<F>, 2> svd;

    // Initialize svd tensor dimensions
    for (UInt n = 0; n < 2; n++) {
      svd[n].dim = std::vector<UInt>(links[n].size() + 1, 0);
      for (UInt i = 0; i < links[n].size(); i++)
        svd[n].dim[i + n] = svd[n].dim[links[n][i]];
    }

    UInt svdSize = Util::multiply(dim);
    UInt dimM = Util::multiply(Util::selectU(dim, links[0]));
    UInt dimN = Util::multiply(Util::selectU(dim, links[1]));

    std::unique_ptr<double[]> svals = std::make_unique<double[]>(options.nsv);
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>((dimM + dimN) * options.nsv);

    double anorm = norm2();

    /* @TODO: Handle case when nvecs != nsv */
    int nvecs =
        Algebra::tensorSVD<F>(dim, links, svals.get(), svecs.get(), data.get(),
                              Algebra::Options(options.nsv, options.tolerance, anorm, Algebra::Target::largest));

    return std::make_tuple(std::move(svd[0]), std::move(svd[1]));
  }

  template <typename F>
  std::vector<std::array<Tensor<F>, 2>> kronecker_SVD(const Tensor<F> &T) {

    std::array<std::vector<UInt>, 2> links{{{0}, {1}}};

    // M = A x B
    auto M = T.matricize({0, 2}, {1, 3});
    // std::cout << "M=" << M << std::endl;

    std::vector<unsigned int> dim = M.dimension();

    UInt dimM = dim[0];
    UInt dimN = dim[1];
    uint dimH = T.dimension()[0];

    std::unique_ptr<F[]> data = std::make_unique<F[]>(M.size());
    std::unique_ptr<double[]> svals = std::make_unique<double[]>(dimM);
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>(dimM * dimM + dimN * dimN);

    M.writeTo(data.get());

    std::cout << "data={";
    for (UInt i = 0; i < M.size(); i++) {
      if (std::abs(data[i]) > 1E-10) {
        std::cout << i << ":" << data[i] << ",";
      }
    }
    std::cout << "}" << std::endl;

    std::cout << "data={";
    for (UInt i = 0; i < dimM; i++) {
      for (UInt j = 0; j < dimN; j++) {
        if (std::abs(M[{i, j}]) > 1E-10) {
          std::cout << "{" << i << "," << j << "}->" << M[{i, j}] << ",";
        }
      }
    }
    std::cout << "}" << std::endl;

    auto opts = Algebra::Options(dimM, Algebra::Target::largest);

    int nvecs = Algebra::tensorSVD(dim, links, svals.get(), svecs.get(), data.get(), opts);

    std::vector<std::array<Tensor<F>, 2>> uv(nvecs);

    std::cout << "svecs={";
    for (UInt i = 0; i < dimM * dimM + dimN * dimN; i++) {
      if (std::abs(svecs[i]) > 1E-10) {
        std::cout << i << ":" << svecs[i] << ",";
      }
    }
    std::cout << "}" << std::endl;

    for (UInt n = 0; n < nvecs; n++) {
      std::cout << "svals[" << n << "]=" << svals[n] << std::endl;
      std::cout << "U[" << n << "]={";
      for (UInt i = 0; i < dimM; i++) {
        if (std::abs(svecs[n * dimM + i]) > 1E-10)
          std::cout << i << ":" << svecs[n * dimM + i] << ",";
      }
      std::cout << "}" << std::endl;
      std::cout << "V[" << n << "]={";
      for (UInt i = 0; i < dimN; i++) {
        if (std::abs(svecs[dimM * dimM + i * dimN + n]) > 1E-10)
          std::cout << i << ":" << svecs[dimM * dimM + i * dimN + n] << ",";
      }
      std::cout << "}" << std::endl;
    }

    for (uint n = 0; n < nvecs; n++) {
      uv[n][0] = Tensor<F>({dimH, dimH});
      for (unsigned int row = 0; row < dim[0]; row++)
        uv[n][0][{row / dimH, row % dimH}] = svecs[n * dim[0] + row] * std::sqrt(svals[n]);

      uv[n][1] = Tensor<F>({dimH, dimH});
      for (unsigned int row = 0; row < dim[1]; row++)
        uv[n][1][{row / dimH, row % dimH}] = svecs[dimM * dim[0] + row * dim[1] + n] * std::sqrt(svals[n]);
    }
    return uv;
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Tensor<F> &T) {
    out << "Tensor: [" << T.subscripts() << "](";
    for (auto const &d : T.dimension())
      out << d << ",";
    out << ")(" << T.size() << "):\n";
    out << "{";
    for (unsigned int i = 0; i < T.size(); i++) {
      if (std::abs(T[i]) > T.tolerance)
        std::cout << i << ":" << T[i] << ",";
    }
    out << "}";
    return out;
  }

} // namespace TNT::Tensor

template class TNT::Tensor::Tensor<double>;
template class TNT::Tensor::Tensor<std::complex<double>>;

template std::vector<std::array<TNT::Tensor::Tensor<double>, 2>> TNT::Tensor::kronecker_SVD(const Tensor<double> &T);
template std::vector<std::array<TNT::Tensor::Tensor<std::complex<double>>, 2>>
TNT::Tensor::kronecker_SVD(const Tensor<std::complex<double>> &T);

template std::ostream &TNT::Tensor::operator<<<double>(std::ostream &, const Tensor<double> &);
template std::ostream &TNT::Tensor::operator<<<std::complex<double>>(std::ostream &,
                                                                     const Tensor<std::complex<double>> &);
