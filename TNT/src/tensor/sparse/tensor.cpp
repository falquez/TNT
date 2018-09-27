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
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

#include <TNT/storage/storage.h>
#include <TNT/tensor/sparse/contraction.h>
#include <TNT/tensor/sparse/tensor.h>

#include "../../algebra/sparse/algebra.h"
#include "../../util/util.h"

namespace TNT::Tensor::Sparse {

  std::vector<UInt> convertIndex(const std::vector<UInt> &stride, const ULong &i) {
    std::vector<UInt> idx(stride.size());
    idx.back() = i / stride.back();
    for (ulong p = 0; p < idx.size() - 1; p++)
      idx[p] = (i % stride[p + 1]) / stride[p];
    return idx;
  }

  ULong convertIndex(const std::vector<UInt> &stride, const std::vector<UInt> &idx) {
    ULong i = 0;
    for (ulong p = 0; p < stride.size(); p++)
      i += idx[p] * stride[p];
    return i;
  }

  template <typename F>
  int writeToFile(const Tensor<F> &tensor, const std::string &filename, const std::string &path) {
    Storage::Storage storage(filename, Storage::FileMode::CreateOverwrite);

    Storage::Data::Metadata<std::vector<unsigned int>> dims{"dimension", tensor.dimension()};

    std::vector<UInt> stride = tensor.strides();
    std::vector<Storage::Data::SparseL<F>> data;
    for (const auto &[idx, v] : tensor.elements()) {
      data.push_back({convertIndex(stride, idx), v});
    }

    std::sort(
        data.begin(), data.end(),
        [](Storage::Data::SparseL<F> a, Storage::Data::SparseL<F> b) { return a.idx < b.idx; });
    storage.create_group(path);
    storage.create(path, dims);

    storage.create(path + "/data", data);

    return 0;
  }

  template <typename F>
  Tensor<F>::Tensor(const std::vector<UInt> dim) : data{}, dim{dim}, stride(dim.size()), sub{} {
    totalDim = 1;
    for (const auto &d : dim)
      totalDim *= d;

    stride[0] = 1;
    for (uint i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];
  }

  template <typename F>
  Tensor<F>::Tensor(const std::string &filename, const std::string &path, const unsigned int &id) {
    Storage::Storage storage(filename, Storage::FileMode::ReadOnly);

    Storage::Data::Metadata<std::vector<unsigned int>> dims{"dimension"};
    storage.read(path, dims);

    std::vector<Storage::Data::SparseL<F>> datalist;
    storage.read(path + "/" + std::to_string(id), datalist);

    dim = dims.value;
    stride.resize(dim.size());
    totalDim = 1;
    for (const auto &d : dim)
      totalDim *= d;
    stride[0] = 1;
    for (uint i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    for (const auto &elem : datalist)
      data.insert({convertIndex(stride, elem.idx), elem.v});
  }

  // template <typename F>
  // F& Tensor<F>::operator[](const std::vector<UInt> &idx){
  //  /*@TODO: check index! */
  //  return data[idx];
  //}

  template <typename F>
  F Tensor<F>::operator[](const std::vector<UInt> &idx) const {
    /*@TODO: check index! */
    typename ConcurrentHashMap<std::vector<UInt>, F>::const_accessor t;
    if (data.find(t, idx))
      return t->second;
    else
      return 0.0;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator<<=(const std::vector<std::tuple<std::vector<UInt>, F>> &v) {
    for (const auto &c : v) {
      assert(std::get<0>(c).size() == dim.size());
      data.insert({std::get<0>(c), std::get<1>(c)});
    }
    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator<<=(const std::tuple<std::vector<UInt>, std::optional<F>> &c) {
    if (std::get<1>(c)) {
      assert(std::get<0>(c).size() == dim.size());
      data.insert({std::get<0>(c), *std::get<1>(c)});
    }
    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator<<=(const std::tuple<ULong, F> &c) {
    assert(std::get<0>(c) < totalDim);
    data.insert({convertIndex(stride, std::get<0>(c)), std::get<1>(c)});
    return *this;
  }

  template <typename F>
  F Tensor<F>::operator[](const ULong &idx) const {
    typename ConcurrentHashMap<std::vector<UInt>, F>::const_accessor t;
    if (data.find(t, convertIndex(stride, idx)))
      return t->second;
    else
      return 0.0;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::merge(const std::string &mrg) {
    std::map<std::string, UInt> idx_map;
    // std::cout << "Tensor<F>::merge(" << mrg << ")" << std::endl;

    std::vector<std::string> mrg_idx = Util::split(mrg, ",");
    std::vector<std::string> sub_idx = Util::split(sub, ",");
    for (UInt i = 0; i < sub_idx.size(); i++)
      idx_map.emplace(sub_idx[i], i);

    assert(mrg_idx.size() == 2 && ((idx_map.at(mrg_idx[1]) - idx_map.at(mrg_idx[0])) == 1));
    UInt m = idx_map.at(mrg_idx[0]);

    std::vector<UInt> ndim(dim.size() - 1);
    std::vector<UInt> nstride(dim.size() - 1);
    for (UInt i = 0; i < m; i++)
      ndim[i] = dim[i];
    ndim[m] = dim[m] * dim[m + 1];
    for (UInt i = m + 1; i < dim.size(); i++)
      ndim[i] = dim[i + 1];

    nstride[0] = 1;
    for (UInt i = 1; i < ndim.size(); i++)
      nstride[i] = ndim[i - 1] * nstride[i - 1];

    ConcurrentHashMap<std::vector<UInt>, F> ndata;
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      std::vector<UInt> nidx(ndim.size());
      std::vector<UInt> idx = it->first;
      for (UInt i = 0; i < m; i++)
        nidx[i] = idx[i];
      nidx[m] = idx[m] + idx[m + 1] * dim[m];
      for (UInt i = m + 1; i < idx.size(); i++)
        nidx[i] = idx[i + 1];
      ndata.insert({nidx, it->second});
    }

    dim = ndim;
    stride = nstride;
    data = ndata;

    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::initialize(const int &mod) {
    for (UInt i = 0; i < totalDim; i++) {
      if ((rand() % mod) == 0) {
        data.insert({convertIndex(stride, i), (rand() % 100) / 100.0});
      }
    }
    return *this;
  }

  template <typename F>
  double Tensor<F>::norm2() const {
    double nrm = 0;
    for (const auto &[idx, v] : data)
      nrm += std::norm(v);
    return std::sqrt(nrm);
  }
  template <typename F>
  Tensor<F> Tensor<F>::conjugate() const {
    Tensor<F> conj(dim);
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      conj.data.insert({it->first, it->second});
    }
    return conj;
  }

  template <>
  Tensor<std::complex<double>> Tensor<std::complex<double>>::conjugate() const {
    Tensor<std::complex<double>> conj(dim);
    for (typename ConcurrentHashMap<std::vector<UInt>, std::complex<double>>::const_iterator it =
             data.begin();
         it != data.end(); it++) {
      conj.data.insert({it->first, std::conj(it->second)});
    }
    return conj;
  }

  template <typename F>
  Tensor<F> Tensor<F>::transpose(const std::vector<UInt> tidx) const {
    assert(tidx.size() == dim.size());
    std::vector<UInt> tdim(dim.size());
    for (uint i = 0; i < dim.size(); i++)
      tdim[i] = dim[tidx[i]];
    Tensor<F> trans(tdim);
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      std::vector<UInt> idx(dim.size());
      for (uint i = 0; i < dim.size(); i++)
        idx[i] = it->first[tidx[i]];
      trans.data.insert({idx, it->second});
    }
    return trans;
  }
  template <typename F>
  Tensor<F> Tensor<F>::matricize(const std::vector<UInt> &idx_r,
                                 const std::vector<UInt> &idx_c) const {
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

    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      std::vector<UInt> idx{0, 0};
      for (uint i = 0; i < idx_r.size(); i++)
        idx[0] += it->first[idx_r[i]] * stride_r[i];
      for (uint i = 0; i < idx_c.size(); i++)
        idx[1] += it->first[idx_c[i]] * stride_c[i];
      mat.data.insert({idx, it->second});
    }

    return mat;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::purge(const double &eps) {
    std::vector<std::vector<UInt>> idxs;
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      if ((std::abs(it->second) <= eps))
        idxs.push_back(it->first);
    }
    for (const auto &idx : idxs)
      data.erase(idx);
    return *this;
  }

  template <typename F>
  std::unordered_map<std::vector<UInt>, F, HashValue<std::vector<UInt>>>
  Tensor<F>::elements() const {
    std::unordered_map<std::vector<UInt>, F, HashValue<std::vector<UInt>>> result;
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      result.emplace(it->first, it->second);
    }
    return result;
  }

  template <typename F>
  std::map<std::vector<UInt>, F> Tensor<F>::elements2() const {
    std::map<std::vector<UInt>, F> result;
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      result.emplace(it->first, it->second);
    }
    return result;
  }

  template <typename F>
  std::map<UInt, F> Tensor<F>::elementsL() const {
    std::map<UInt, F> result;
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      result.emplace(convertIndex(stride, it->first), it->second);
    }
    return result;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::readFrom(const F *source, const double &eps) {
    for (ULong i = 0; i < totalDim; i++) {
      if (std::abs(source[i]) > EPS) {
        data.insert({convertIndex(stride, i), source[i]});
      }
    }
    return *this;
  }

  template <typename F>
  const Tensor<F> &Tensor<F>::writeTo(F *target, const double &eps) const {
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      target[convertIndex(stride, it->first)] = it->second;
    }
    return *this;
  }

  template <typename F>
  Tensor<F> &Tensor<F>::operator=(const Contraction<F> &tc) {
    int err = 0;
    int prev = 0, curr = 0;

    std::vector<std::string> index = Util::split(sub, ",");
    dim = std::vector<UInt>(index.size());
    stride = std::vector<UInt>(index.size());

    for (unsigned int i = 0; i < dim.size(); i++) {
      dim[i] = tc.dim_map.at(index[i]);
    }

    stride[0] = 1;
    for (uint i = 1; i < dim.size(); i++)
      stride[i] = dim[i - 1] * stride[i - 1];

    totalDim = 1;
    for (const auto &d : dim)
      totalDim *= d;

    std::string result_sub;
    std::array<std::vector<std::string>, 2> idx;
    std::array<Tensor<F>, 2> T;
    std::array<const Tensor<F> *, 2> tupleT;

    idx[0] = Util::split(tc.tensors[0]->sub, ",");
    std::sort(idx[0].begin(), idx[0].end());
    tupleT = {tc.tensors[0], tc.tensors[1]};

    for (unsigned int i = 1; i < tc.tensors.size(); i++) {
      std::vector<std::string> result_idx;

      prev = (i - 1) % 2;
      curr = i % 2;

      idx[curr] = Util::split(tc.tensors[i]->sub, ",");
      std::sort(idx[curr].begin(), idx[curr].end());

      if (i != tc.tensors.size() - 1) {
        std::set_symmetric_difference(idx[prev].begin(), idx[prev].end(), idx[curr].begin(),
                                      idx[curr].end(), std::back_inserter(result_idx));

        result_sub = Util::concat(result_idx, ",");
      } else {
        result_sub = sub;
      }

      T[curr] = contract<F>(result_sub, tc.dim_map, tupleT);
      idx[curr] = result_idx;
      T[curr].sub = result_sub;
      tupleT = {&T[curr], tc.tensors[i + 1]};
    }

    data = std::move(T[curr].data);

    // clear subscripts
    sub.clear();
    return *this;
  }

  template <typename F>
  Tensor<F> Tensor<F>::normalize_QRD(const UInt &idx_r, const UInt &svdn) {
    Tensor<F> R({dim[idx_r], dim[idx_r]});
    std::array<std::unique_ptr<F[]>, 2> buff;

    std::array<std::vector<unsigned int>, 2> links;
    for (unsigned int i = 0; i < dim.size(); i++)
      if (i != idx_r)
        links[0].push_back(i);
    links[1] = {idx_r};

    buff[0] = std::make_unique<F[]>(totalDim);
    buff[1] = std::make_unique<F[]>(R.totalDim);

    writeTo(buff[0].get());

    int res = Algebra::Sparse::tensorQRD(dim, links, buff[0].get(), buff[1].get());

    data.clear();
    readFrom(buff[0].get());
    R.readFrom(buff[1].get());

    return R;
  }

  template <typename F>
  std::tuple<Tensor<F>, Tensor<F>> Tensor<F>::SVD(std::array<std::string, 2> subscript,
                                                  const SVDOptions &options) const {
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
    std::set_intersection(subs[0].begin(), subs[0].end(), subs[1].begin(), subs[1].end(),
                          std::back_inserter(svd_sub));
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

    std::unique_ptr<double[]> svals = std::make_unique<double[]>(options.nsv);
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>((dimM + dimN) * options.nsv);
    std::unique_ptr<F[]> lvecs = std::make_unique<F[]>(dimM * options.nsv);
    std::unique_ptr<F[]> rvecs = std::make_unique<F[]>(dimN * options.nsv);

    double anorm = norm2();

    UInt nvecs = Algebra::Sparse::tensorSVD<F>(
        *this, links, svals.get(), svecs.get(),
        Algebra::Options(options.nsv, options.tolerance, anorm, Algebra::Target::largest));

    switch (options.norm) {
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
      links_tr[n].insert(links_tr[n].begin() + std::distance(subs[n].begin(), iter),
                         links[n].size());
    }

    std::array<std::unique_ptr<F[]>, 2> buffer;
    buffer[0] = std::make_unique<F[]>(dimM * options.nsv);
    buffer[1] = std::make_unique<F[]>(dimN * options.nsv);

    err = Algebra::transpose(dim_tr[0], links_tr[0], lvecs.get(), buffer[0].get());
    err = Algebra::transpose(dim_tr[1], links_tr[1], rvecs.get(), buffer[1].get());

    svd[0].readFrom(buffer[0].get());
    svd[1].readFrom(buffer[1].get());

    return std::make_tuple(std::move(svd[0]), std::move(svd[1]));
  }

  template <typename F>
  std::tuple<Tensor<F>, Tensor<F>> Tensor<F>::SVD(std::array<std::vector<UInt>, 2> links, UInt nsv,
                                                  SVDNorm norm) const {
    int err = 0;
    std::array<Tensor<F>, 2> svd;

    return std::make_tuple(std::move(svd[0]), std::move(svd[1]));
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator+(const Tensor<F> &M) const {
    Tensor<F> T = M;

    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      typename ConcurrentHashMap<std::vector<UInt>, F>::accessor t;
      T.data.insert(t, it->first);
      t->second += it->second;
    }
    return T;
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator-() const {
    Tensor<F> T(dim);

    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      typename ConcurrentHashMap<std::vector<UInt>, F>::accessor t;
      T.data.insert(t, it->first);
      t->second = -it->second;
    }
    return T;
  }

  template <typename F>
  Tensor<F> Tensor<F>::operator*(const F &c) const {
    Tensor<F> M(dim);
    for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it = data.begin();
         it != data.end(); it++) {
      typename ConcurrentHashMap<std::vector<UInt>, F>::accessor t;
      M.data.insert(t, it->first);
      t->second = it->second * c;
    }
    return M;
  }

  template <typename F>
  std::vector<std::array<Tensor<F>, 2>> kronecker_SVD(const Tensor<F> &T, UInt nsv,
                                                      const double tolerance) {
    std::vector<std::array<Tensor<F>, 2>> uv(nsv);

    // M = A x B
    auto M = T.matricize({0, 2}, {1, 3});
    // std::cout << "M=" << M << std::endl;

    std::vector<unsigned int> dim = M.dimension();

    std::unique_ptr<F[]> data = std::make_unique<F[]>(M.size());
    std::unique_ptr<F[]> svecs = std::make_unique<F[]>(nsv * (dim[0] + dim[1]));
    std::unique_ptr<double[]> svals = std::make_unique<double[]>(nsv);

    M.writeTo(data.get());
    double aNorm = M.norm2();

    int nvecs =
        Algebra::tensorSVD(dim, {{{0}, {1}}}, svals.get(), svecs.get(), data.get(),
                           Algebra::Options(nsv, tolerance, aNorm, Algebra::Target::largest));

    uint dimH = T.dimension()[0];
    for (uint n = 0; n < nvecs; n++) {
      uv[n][0] = Tensor<F>({dimH, dimH});
      uv[n][1] = Tensor<F>({dimH, dimH});
      // if(std::abs(svals[n]) > Vector::EPS){
      for (unsigned int row = 0; row < dim[0]; row++)
        uv[n][0] <<= {{row / dimH, row % dimH}, svecs[n * dim[0] + row] * sqrt(svals[n])};
      // uv[n][0] <<= {{row%dimH,row/dimH},svecs[n*dim[0]+row]*sqrt(svals[n])};
      for (unsigned int row = 0; row < dim[1]; row++)
        uv[n][1] <<=
            {{row / dimH, row % dimH}, svecs[nvecs * dim[0] + n * dim[1] + row] * sqrt(svals[n])};
      // uv[n][1] <<= {{row%dimH,row/dimH},svecs[nvecs*dim[0]+
      // n*dim[1]+row]*sqrt(svals[n])};
      uv[n][0].purge(tolerance);
      uv[n][1].purge(tolerance);
      //}
    }
    return uv;
  }

  template <typename F>
  Tensor<F> IdentityMatrix(unsigned int d) {
    Tensor<F> Id({d, d});
    for (unsigned int i = 0; i < d; i++) {
      Id <<= {{i, i}, 1.0};
    }
    return Id;
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Tensor<F> &T) {
    auto elements = T.elements2();
    out << "Tensor: [" << T.subscripts() << "] d=(";
    for (auto const &d : T.dimension())
      out << d << ",";
    out << ") s=(";
    for (auto const &d : T.strides())
      out << d << ",";
    std::cout << ") (" << elements.size() << "):" << std::endl;
    out << "{";
    for (const auto &[idx, v] : elements) {
      std::cout << "{";
      for (auto i : idx)
        std::cout << i + 1 << ",";
      std::cout << "}->" << v << ", ";
    }
    out << "}";
    return out;
  }

} // namespace TNT::Tensor::Sparse

template class TNT::Tensor::Sparse::Tensor<double>;
template class TNT::Tensor::Sparse::Tensor<std::complex<double>>;

template std::vector<std::array<TNT::Tensor::Sparse::Tensor<double>, 2>>
TNT::Tensor::Sparse::kronecker_SVD<double>(const Tensor<double> &, UInt, const double);
template std::vector<std::array<TNT::Tensor::Sparse::Tensor<std::complex<double>>, 2>>
TNT::Tensor::Sparse::kronecker_SVD<std::complex<double>>(const Tensor<std::complex<double>> &, UInt,
                                                         const double);

template int TNT::Tensor::Sparse::writeToFile<double>(const Tensor<double> &tensor,
                                                      const std::string &filename,
                                                      const std::string &path);
template int
TNT::Tensor::Sparse::writeToFile<std::complex<double>>(const Tensor<std::complex<double>> &tensor,
                                                       const std::string &filename,
                                                       const std::string &path);

template TNT::Tensor::Sparse::Tensor<double> TNT::Tensor::Sparse::IdentityMatrix(unsigned int d);
template TNT::Tensor::Sparse::Tensor<std::complex<double>>
TNT::Tensor::Sparse::IdentityMatrix(unsigned int d);

template std::ostream &TNT::Tensor::Sparse::operator<<<double>(std::ostream &out,
                                                               const Tensor<double> &T);
template std::ostream &TNT::Tensor::Sparse::
operator<<<std::complex<double>>(std::ostream &out, const Tensor<std::complex<double>> &T);
