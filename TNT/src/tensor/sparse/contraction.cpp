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
#include <complex>
#include <iostream>

#include <TNT/tensor/sparse/contraction.h>
#include <TNT/tensor/sparse/tensor.h>

#include "../../util/util.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace TNT::Tensor::Sparse {

  template <typename F>
  class ConcurrentHashContraction {

    const ConcurrentHashMap<std::vector<UInt>, F> &factor;
    ConcurrentHashMap<std::vector<UInt>, F> &result;

    std::vector<UInt> pos;
    std::vector<UInt> free_idx;

  public:
    ConcurrentHashContraction(const ConcurrentHashMap<std::vector<UInt>, F> &factor,
                              ConcurrentHashMap<std::vector<UInt>, F> &result, std::vector<UInt> pos,
                              std::vector<UInt> fidx)
        : factor(factor), result(result), pos(pos), free_idx(fidx) {}

    void operator()(typename ConcurrentHashMap<std::vector<UInt>, F>::const_range_type &r) const {

      for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it1 = r.begin(); it1 != r.end(); ++it1) {
        std::vector<UInt> ind = it1->first;
        F c1 = it1->second;
        std::vector<UInt> key(pos.size());
        std::vector<UInt> r_idx(free_idx.size());
        for (unsigned int i = 0; i < pos.size(); i++)
          key[i] = ind[pos[i]];
        for (unsigned int i = 0; i < free_idx.size(); i++)
          r_idx[i] = ind[free_idx[i]];

        typename ConcurrentHashMap<std::vector<UInt>, F>::const_accessor t2;
        if (factor.find(t2, key)) {
          typename ConcurrentHashMap<std::vector<UInt>, F>::accessor t1;
          result.insert(t1, r_idx);
          t1->second += t2->second * c1;
        }
      }
    }
  };

  template <typename F>
  class ConcurrentEmplace {

    ConcurrentMultiMap<std::vector<UInt>, std::vector<UInt>> &elements;
    std::vector<UInt> pos;

  public:
    ConcurrentEmplace(ConcurrentMultiMap<std::vector<UInt>, std::vector<UInt>> &elem, std::vector<UInt> _pos)
        : elements(elem), pos(_pos) {}

    void operator()(typename ConcurrentHashMap<std::vector<UInt>, F>::const_range_type &r) const {
      for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it1 = r.begin(); it1 != r.end(); ++it1) {
        std::vector<UInt> ind = it1->first;
        std::vector<UInt> key(pos.size());
        for (unsigned int i = 0; i < pos.size(); i++)
          key[i] = ind[pos[i]];
        elements.emplace(key, ind);
      }
    }
  };

  template <typename F>
  class ConcurrentUpdate {

    ConcurrentMultiMap<std::vector<UInt>, std::vector<UInt>> &elements;
    const ConcurrentHashMap<std::vector<UInt>, F> &factor;
    ConcurrentMultiMap<std::vector<UInt>, std::array<std::vector<UInt>, 2>> &elements2;

    std::vector<UInt> pos;
    std::vector<std::array<UInt, 2>> free_idx;

  public:
    ConcurrentUpdate(ConcurrentMultiMap<std::vector<UInt>, std::vector<UInt>> &elements,
                     const ConcurrentHashMap<std::vector<UInt>, F> &factor,
                     ConcurrentMultiMap<std::vector<UInt>, std::array<std::vector<UInt>, 2>> &elements2,
                     std::vector<UInt> pos, std::vector<std::array<UInt, 2>> fidx)
        : elements(elements), factor(factor), elements2(elements2), pos(pos), free_idx(fidx) {}

    void operator()(typename ConcurrentHashMap<std::vector<UInt>, F>::const_range_type &r) const {
      for (typename ConcurrentHashMap<std::vector<UInt>, F>::const_iterator it1 = r.begin(); it1 != r.end(); ++it1) {
        std::vector<UInt> ind1 = it1->first;

        std::vector<UInt> key(pos.size());
        for (unsigned int i = 0; i < pos.size(); i++)
          key[i] = ind1[pos[i]];

        // Loop over all matching tensor[0] elements
        auto range = elements.equal_range(key);
        for (auto it0 = range.first; it0 != range.second; ++it0) {
          std::array<std::vector<UInt>, 2> ind;
          ind[0] = it0->second;
          ind[1] = ind1;
          std::vector<UInt> r_idx(free_idx.size());
          for (unsigned int i = 0; i < r_idx.size(); i++) {
            r_idx[i] = ind[free_idx[i][0]][free_idx[i][1]];
          }
          elements2.emplace(r_idx, ind);
        }
      }
    }
  };

  template <typename F>
  class ConcurrentSum {
    const ConcurrentHashMap<std::vector<UInt>, F> &factor0;
    const ConcurrentHashMap<std::vector<UInt>, F> &factor1;
    ConcurrentHashMap<std::vector<UInt>, F> &result;

  public:
    ConcurrentSum(const ConcurrentHashMap<std::vector<UInt>, F> &factor0,
                  const ConcurrentHashMap<std::vector<UInt>, F> &factor1,
                  ConcurrentHashMap<std::vector<UInt>, F> &result)
        : factor0(factor0), factor1(factor1), result(result) {}

    void operator()(
        typename ConcurrentMultiMap<std::vector<UInt>, std::array<std::vector<UInt>, 2>>::const_range_type &r) const {
      for (typename ConcurrentMultiMap<std::vector<UInt>, std::array<std::vector<UInt>, 2>>::const_iterator it =
               r.begin();
           it != r.end(); ++it) {
        std::vector<UInt> r_idx = it->first;
        std::array<std::vector<UInt>, 2> ind = it->second;
        typename ConcurrentHashMap<std::vector<UInt>, F>::const_accessor t0, t1;
        if (factor0.find(t0, ind[0]) && factor1.find(t1, ind[1])) {
          typename ConcurrentHashMap<std::vector<UInt>, F>::accessor tr;
          result.insert(tr, r_idx);
          tr->second += (t0->second) * (t1->second);
        }
      }
    }
  };

  template <typename F>
  Tensor<F> contract2(const std::string subscript_r, std::vector<UInt> dim_r,
                      const std::array<const Tensor<F> *, 2> &t) {
    Tensor<F> result(dim_r);

    std::array<std::string, 2> subs;
    std::array<std::vector<std::string>, 2> idx;
    std::vector<unsigned int> free_idx;

    // Tensor subscripts
    subs[0] = t[0]->sub;
    subs[1] = t[1]->sub;

    // Read tensor indices
    idx[0] = Util::split(t[0]->sub, ",");
    idx[1] = Util::split(t[1]->sub, ",");

    // std::cout << "contract2:" << subscript_r << "<-" << subs[0] << "*" <<
    // subs[1] <<"\n";

    std::vector<std::string> result_idx = Util::split(subscript_r, ",");
    for (const auto &c : result_idx) {
      unsigned int n = 0;
      auto iter = std::find(idx[n].begin(), idx[n].end(), c);
      unsigned int l = std::distance(idx[n].begin(), iter);
      if (iter != idx[n].end())
        free_idx.push_back(l);
    }

    std::vector<UInt> kpos;
    for (const auto &c : idx[1]) {
      auto iter = std::find(idx[0].begin(), idx[0].end(), c);
      unsigned int l = std::distance(idx[0].begin(), iter);
      if (iter != idx[0].end())
        kpos.push_back(l);
    }

    typename ConcurrentHashMap<std::vector<UInt>, F>::const_range_type r = t[0]->data.range();

    tbb::parallel_for(r, ConcurrentHashContraction<F>(t[1]->data, result.data, kpos, free_idx),
                      tbb::auto_partitioner());

    return std::move(result);
  }

  template <typename F>
  Tensor<F> contract(const std::string subscript, std::map<std::string, UInt> dim_map,
                     const std::array<const Tensor<F> *, 2> &t) {
    std::array<std::string, 2> subs;
    std::array<std::vector<std::string>, 2> idx;
    std::array<std::vector<UInt>, 2> pos;
    std::vector<std::string> contraction_idx;
    std::vector<std::array<UInt, 2>> free_idx;
    std::vector<UInt> rdim;

    // std::cout << "Sparse::contract()" << std::endl;

    // Tensor subscripts
    subs[0] = t[0]->sub;
    subs[1] = t[1]->sub;

    // Read contraction indices
    idx[0] = Util::split(subs[0], ",");
    idx[1] = Util::split(subs[1], ",");
    std::sort(idx[0].begin(), idx[0].end());
    std::sort(idx[1].begin(), idx[1].end());
    std::set_intersection(idx[0].begin(), idx[0].end(), idx[1].begin(), idx[1].end(),
                          std::back_inserter(contraction_idx));

    // Read tensor indices
    idx[0] = Util::split(t[0]->sub, ",");
    idx[1] = Util::split(t[1]->sub, ",");

    std::vector<std::string> result_idx = Util::split(subscript, ",");
    for (const auto &c : result_idx) {
      rdim.push_back(dim_map.at(c));
      for (unsigned int n = 0; n < 2; n++) {
        auto iter = std::find(idx[n].begin(), idx[n].end(), c);
        unsigned int l = std::distance(idx[n].begin(), iter);
        if (iter != idx[n].end())
          free_idx.push_back({n, l});
      }
    }

    // Find the position of contracted indices in the tensor indices string
    for (const auto &c : contraction_idx) {
      for (unsigned int n = 0; n < 2; n++) {
        auto iter = std::find(idx[n].begin(), idx[n].end(), c);
        unsigned int l = std::distance(idx[n].begin(), iter);
        if (iter != idx[n].end())
          pos[n].push_back(l);
      }
    }

    // Partition tensor[0] elements according to its contracted indices
    ConcurrentMultiMap<std::vector<UInt>, std::vector<UInt>> elements;
    ConcurrentMultiMap<std::vector<UInt>, std::array<std::vector<UInt>, 2>> elements2;

    typename ConcurrentHashMap<std::vector<UInt>, F>::const_range_type r1 = t[0]->data.range();
    tbb::parallel_for(r1, ConcurrentEmplace<F>(elements, pos[0]), tbb::auto_partitioner());

    typename ConcurrentHashMap<std::vector<UInt>, F>::const_range_type r2 = t[1]->data.range();
    tbb::parallel_for(r2, ConcurrentUpdate<F>(elements, t[0]->data, elements2, pos[1], free_idx),
                      tbb::auto_partitioner());

    Tensor<F> result(rdim);
    typename ConcurrentMultiMap<std::vector<UInt>, std::array<std::vector<UInt>, 2>>::const_range_type r3 =
        elements2.range();
    tbb::parallel_for(r3, ConcurrentSum<F>(t[0]->data, t[1]->data, result.data), tbb::auto_partitioner());

    return std::move(result);
  }

  template <typename F>
  Contraction<F>::Contraction(const Tensor<F> &t)
      : dims{t.dim}, strides{t.stride}, subs{t.sub}, tensors{&t}, dim_map{} {

    std::vector<std::string> idx = Util::split(subs[0], ",");

    for (unsigned int i = 0; i < dims[0].size(); i++)
      dim_map.emplace(idx[i], dims[0][i]);
  }

  template <typename F>
  Contraction<F> &Contraction<F>::operator*(const Tensor<F> &t) {

    tensors.push_back(&t);
    subs.push_back(t.sub);
    dims.push_back(t.dim);
    strides.push_back(t.stride);

    std::vector<std::string> idx = Util::split(t.sub, ",");
    for (unsigned int i = 0; i < t.dim.size(); i++)
      dim_map.emplace(idx[i], t.dim[i]);

    return *this;
  }

  template <typename F>
  F Contraction<F>::dotProduct() const {
    throw std::invalid_argument("Not Implemented");

    // return TNT::BLAS::dot<F>(totalDim[0], {data[0], data[1]});
  }
} // namespace TNT::Tensor::Sparse

template class TNT::Tensor::Sparse::Contraction<double>;
template class TNT::Tensor::Sparse::Contraction<std::complex<double>>;

template TNT::Tensor::Sparse::Tensor<double>
TNT::Tensor::Sparse::contract2<double>(const std::string subscript_r, std::vector<UInt> dim_r,
                                       const std::array<const Tensor<double> *, 2> &t);
template TNT::Tensor::Sparse::Tensor<std::complex<double>>
TNT::Tensor::Sparse::contract2<std::complex<double>>(const std::string subscript_r, std::vector<UInt> dim_r,
                                                     const std::array<const Tensor<std::complex<double>> *, 2> &t);

template TNT::Tensor::Sparse::Tensor<double>
TNT::Tensor::Sparse::contract<double>(const std::string subscript, std::map<std::string, UInt> dim_map,
                                      const std::array<const Tensor<double> *, 2> &t);
template TNT::Tensor::Sparse::Tensor<std::complex<double>>
TNT::Tensor::Sparse::contract<std::complex<double>>(const std::string subscript, std::map<std::string, UInt> dim_map,
                                                    const std::array<const Tensor<std::complex<double>> *, 2> &t);
