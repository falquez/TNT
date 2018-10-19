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

#ifndef _TNT_TENSOR_TENSOR_H
#define _TNT_TENSOR_TENSOR_H

#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

//#include <TNT/tensor/sparse/tensor.h>

namespace TNT::Tensor {

  using UInt = unsigned int;
  using ULong = unsigned long long;
  const double EPS = 10E-12;

  enum SVDNorm { equal, left, right };
  struct SVDOptions {
    SVDNorm norm = SVDNorm::equal;
    UInt nsv = 0;
    double tolerance = 1e-9;
  };

  template <typename F>
  class Tensor;
  template <typename F>
  class Contraction;
  template <typename F>
  class EigenSolver;

  template <typename F>
  using TensorScalar = std::tuple<Tensor<F>, F>;

  template <typename F>
  class Tensor {

    std::unique_ptr<F[]> data;
    std::vector<UInt> dim;
    std::vector<UInt> stride;
    mutable std::string sub;
    UInt totalDim;

    friend class Contraction<F>;
    friend class EigenSolver<F>;
    // friend class Sparse::EigenSolver<F>;

  public:
    Tensor() : data{nullptr}, dim{}, stride{}, sub{}, totalDim{0} {}
    Tensor(const std::vector<UInt> &dim);
    Tensor(const std::vector<UInt> &dim, const F initial);
    // Tensor(const Sparse::Tensor<F> &t);
    Tensor(const std::string &filename, const std::string &path, const unsigned int &id = 0);
    Tensor<F>(Tensor<F> &&t) = default;
    Tensor<F>(const Tensor<F> &t);
    Tensor<F>(std::tuple<std::vector<UInt>, std::unique_ptr<F[]>> &&data);
    Tensor<F> &operator=(const Tensor<F> &t);

    Tensor<F> &operator=(Tensor<F> &&t) = default;

    Tensor<F> &operator=(const Contraction<F> &tc);

    Tensor<F> &operator*=(const Tensor<F> &M);

    Tensor<F> operator+(const Tensor<F> &M) const;
    Tensor<F> operator-(const Tensor<F> &M) const;
    Tensor<F> operator-() const;
    Tensor<F> operator*(const F &c) const;
    Contraction<F> operator*(const Tensor<F> &rhs) const;

    int writeToFile(const std::string &filename, const std::string &path = "/Tensor", const unsigned int &id = 0) const;

    F &operator[](const std::vector<UInt> &idx) {
      int k = 0;
      for (UInt i = 0; i < stride.size(); i++)
        k = k + idx[i] * stride[i];
      return data[k];
    }

    F operator[](const std::vector<UInt> &idx) const {
      UInt k = 0;
      for (UInt i = 0; i < stride.size(); i++)
        k = k + idx[i] * stride[i];
      return data[k];
    }

    F &operator[](const UInt i) { return data[i]; }

    F operator[](const UInt i) const { return data[i]; }

    std::vector<UInt> dimension() const { return dim; }
    std::vector<UInt> strides() const { return stride; }

    UInt dimension(const std::string &s) const;

    UInt size() const { return totalDim; }

    std::string subscripts() const { return sub; }

    Tensor<F> &operator()(const std::string &idx) {
      sub = idx;
      return *this;
    }

    const Tensor<F> &operator()(const std::string &idx) const {
      sub = idx;
      return *this;
    }

    Tensor<F> expand(const std::string &eidx, const UInt &edim, bool initialize = true, const int &mod = 7) const;

    Tensor<F> &merge(const std::string &idx);
    Tensor<F> &split(const std::string &idx);

    Tensor<F> &merge(const std::array<UInt, 3> &idx);
    Tensor<F> &split(const std::array<UInt, 3> &idx);

    Tensor<F> trace(const std::string &idx) const;

    double norm2() const;

    // Sparse::Tensor<F> sparse();

    Tensor<F> &initialize(const int &mod = 1);

    Tensor<F> conjugate() const;

    Tensor<F> &readFrom(const F *source, const double &eps = EPS);
    const Tensor<F> &writeTo(F *target, const double &eps = EPS) const;
    const Tensor<F> &addTo(F *target, const F &alpha = 1.0, const double &eps = EPS) const;

    // Calculate M_{abc} -> M_{asc} R_{bs}, so that M_{abc}M*_{ab'c} = delta_{bb'}
    // Return R_{bs}
    Tensor<F> normalize_QRD(const UInt &idx_r);

    // Calculate M_{abc} -> M_{asc}R_{sb} , so that M_{abc}M*_{ab'c} = delta_{bb'}
    // Return M_{asc} R_{sb}
    std::tuple<Tensor<F>, Tensor<F>> normalize_QRD(const std::string &idx_s);

    /* SVD between links[0] and links[1]
     *     auto [Bu,Bv] = B("a1,s,s',a3").SVD({{{0,1},{2,3}}});
     *     B2("a1,s,s',a3") = Bu("a1,s,b")*Bv("b,s',a3");
     * then B == B2 up to numerical error
     */
    std::tuple<Tensor<F>, Tensor<F>> SVD(std::array<std::string, 2> subscript,
                                         const SVDOptions &options = SVDOptions{}) const;

    std::tuple<Tensor<F>, Tensor<F>> SVD(std::array<std::string, 2> subscript, const Tensor<F> &left,
                                         const Tensor<F> &right, const SVDOptions &options = SVDOptions{}) const;

    /* SVD between links[0] and links[1]
     *     auto [Bu,Bv] = B("a1,s,s',a3").SVD({{"s,a1,b","s',b,a3"}});
     *     B2("a1,s,s',a3") = Bu("s,a1,b")*Bv("s',b,a3");
     * then B == B2 up to numerical error
     * NOTE: currently supports only 1 SVD index.
     */
    std::tuple<Tensor<F>, Tensor<F>> SVD(std::array<std::vector<UInt>, 2> links,
                                         const SVDOptions &options = SVDOptions{}) const;
  };

  template <typename F = double>
  std::ostream &operator<<(std::ostream &out, const Tensor<F> &T);
} // namespace TNT::Tensor

#endif // _TNT_TENSOR_TENSOR_H
