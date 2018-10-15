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

#ifndef _TNT_TENSOR_SPARSE_TENSOR_H
#define _TNT_TENSOR_SPARSE_TENSOR_H

#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <TNT/tensor/vector.h>
#include <boost/functional/hash.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>

namespace TNT::Tensor {
  using UInt = unsigned int;
  using ULong = unsigned long long;

  enum SVDNorm { equal, left, right };
  struct SVDOptions {
    SVDNorm norm = SVDNorm::equal;
    UInt nsv = 0;
    double tolerance = 1e-9;
  };
} // namespace TNT::Tensor

namespace TNT::Tensor::Sparse {
  using UInt = unsigned int;
  using ULong = unsigned long long;

  const double EPS = 10E-12;

  template <typename U>
  struct HashValue {
    size_t operator()(const U &p) const { return boost::hash_value(p); }
  };

  template <typename U>
  struct HashCompare {
    static size_t hash(const U &p) { return boost::hash_value(p); }
    static bool equal(const U &x, const U &y) { return x == y; }
  };

  template <typename U, typename F>
  using ConcurrentHashMap = tbb::concurrent_hash_map<U, F, HashCompare<U>>;

  template <typename U, typename F>
  using ConcurrentMap = tbb::concurrent_unordered_map<U, F, HashValue<U>>;

  template <typename U, typename F>
  using ConcurrentMultiMap = tbb::concurrent_unordered_multimap<U, F, HashValue<U>>;

  template <typename F>
  class Contraction;
  template <typename F>
  class EigenSolver;
  template <typename F>
  class Tensor;

  template <typename F>
  using TensorConstraint = std::tuple<Tensor<F>, F, F>;

  template <typename F>
  Tensor<F> contract(const std::string subscript, std::map<std::string, UInt> dim_map,
                     const std::array<const Tensor<F> *, 2> &t);

  template <typename F>
  Tensor<F> contract2(const std::string subscript_r, std::vector<UInt> dim_r,
                      const std::array<const Tensor<F> *, 2> &t);

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Tensor<F> &T);

  // template <typename F>
  // int writeToFile(const Tensor<F> &tensor, const std::string &filename, const std::string &path = "/Tensor/Sparse");

  template <typename F>
  class Tensor {
    ConcurrentHashMap<std::vector<UInt>, F> data;
    std::vector<UInt> dim;
    std::vector<UInt> stride;
    ULong totalDim;
    mutable std::string sub;
    mutable std::vector<UInt> curr_idx;

    friend class Contraction<F>;
    friend class EigenSolver<F>;

  public:
    Tensor() : data{}, dim{}, stride{}, sub{}, totalDim{0} {}
    Tensor(const UInt &shape) : data{}, dim(shape), stride(shape), sub{}, totalDim{0} {}

    Tensor(const std::vector<UInt> dim);

    Tensor(const std::string &filename, const std::string &path, const unsigned int &id = 0);
    //~Tensor(){}

    int writeToFile(const std::string &filename, const std::string &path = "/Tensor/Sparse",
		    const unsigned int &id = 0);

    Tensor<F> &operator<<=(const std::vector<std::tuple<std::vector<UInt>, F>> &c);
    Tensor<F> &operator<<=(const std::tuple<std::vector<UInt>, std::optional<F>> &c);
    Tensor<F> &operator<<=(const std::tuple<ULong, F> &c);
    Tensor<F> &operator+=(const std::tuple<std::vector<UInt>, std::optional<F>> &c);

    F operator[](const std::vector<UInt> &idx) const;

    F operator[](const ULong &idx) const;

    std::unordered_map<std::vector<UInt>, F, HashValue<std::vector<UInt>>> elements() const;
    std::map<std::vector<UInt>, F> elements2() const;
    std::map<UInt, F> elementsL() const;

    double norm2() const;

    std::vector<UInt> dimension() const { return dim; }

    std::vector<UInt> strides() const { return stride; }

    ULong size() const { return totalDim; }

    ULong dataSize() const { return data.size(); }

    std::string subscripts() const { return sub; }

    Tensor<F> &operator()(const std::string &subscript) {
      sub = subscript;
      return *this;
    }

    const Tensor<F> &operator()(const std::string &subscript) const {
      sub = subscript;
      return *this;
    }

    Tensor<F> &merge(const std::string &idx);
    Tensor<F> &split(const std::string &idx);

    Tensor<F> &merge(const std::array<UInt, 3> &idx);
    Tensor<F> &split(const std::array<UInt, 3> &idx);

    Tensor<F> trace(const std::string &idx) const;

    Tensor<F> &readFrom(const F *source, const double &eps = EPS);
    const Tensor<F> &writeTo(F *target, const double &eps = EPS) const;
    const Tensor<F> &addTo(F *target, const F alpha, const double &eps = EPS) const;

    Tensor<F> &purge(const double &eps = EPS);

    Tensor<F> &operator=(const Tensor<F> &t);
    // Tensor<F> &operator=(Tensor<F> &&t) = default;
    Tensor<F> &operator=(const Contraction<F> &tc);

    Tensor<F> &operator+=(const Tensor<F> &M);
    Tensor<F> operator+(const Tensor<F> &M) const;
    Tensor<F> operator-(const Tensor<F> &M) const;

    Tensor<F> transpose(const std::vector<UInt> tidx) const;
    Tensor<F> matricize(const std::vector<UInt> &row_idx, const std::vector<UInt> &col_idx) const;

    Tensor<F> operator*(const F &c) const;
    Tensor<F> operator-() const;

    Contraction<F> operator*(const Tensor<F> &rhs) const { return Contraction<F>(*this) * rhs; }

    Tensor<F> &initialize(const int &mod = 701);

    Tensor<F> conjugate() const;

    // Calculate M_{abc} -> M_{asc} R_{sb}, so that M_{abc}M*_{ab'c} = delta_{bb'}
    // Return R_{bs}
    Tensor<F> normalize_QRD(const UInt &idx_r, const UInt &svdn = 0);

    /* SVD between links[0] and links[1]
     *     auto [Bu,Bv] = B("a1,s,s',a3").SVD({{{0,1},{2,3}}});
     *     B2("a1,s,s',a3") = Bu("a1,s,b")*Bv("b,s',a3");
     * then B == B2 up to numerical error
     */
    std::tuple<Tensor<F>, Tensor<F>> SVD(std::array<std::string, 2> subscript,
                                         const SVDOptions &options = SVDOptions{}) const;
    // std::tuple<Tensor<F>, Tensor<F>> SVD(std::array<std::string, 2> subscript,
    // const SVDOptions &options = SVDOptions{}) const;

    /* SVD between links[0] and links[1]
     *     auto [Bu,Bv] = B("a1,s,s',a3").SVD({{"s,a1,b","s',b,a3"}});
     *     B2("a1,s,s',a3") = Bu("s,a1,b")*Bv("s',b,a3");
     * then B == B2 up to numerical error
     * NOTE: currently supports only 1 SVD index.
     */
    std::tuple<Tensor<F>, Tensor<F>> SVD(std::array<std::vector<UInt>, 2> links, UInt nsv = 0,
                                         SVDNorm norm = SVDNorm::equal) const;

    friend Tensor<F> contract<>(const std::string subscript, std::map<std::string, UInt> dim_map,
                                const std::array<const Tensor<F> *, 2> &t);
    friend Tensor<F> contract2<>(const std::string subscript_r, std::vector<UInt> dim_r,
                                 const std::array<const Tensor<F> *, 2> &t);
  };

  template <typename F>
  Tensor<F> IdentityMatrix(UInt d);

  template <typename F>
  std::vector<std::array<Tensor<F>, 2>> kronecker_SVD(const Tensor<F> &T, UInt nsv, const double tolerance = EPS);

} // namespace TNT::Tensor::Sparse

#endif // _TNT_TENSOR_SPARSE_TENSOR_H
