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

#include <TNT/operator/mpo.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/sparse/tensor.h>
#include <TNT/tensor/tensor.h>

#include "../algebra/algebra.h"
#include "../parser/parser.h"

namespace TNT::Operator {

  /*
   *        3
   *     +--^--+
   *  0->|  W  |->1
   *     +--^--+
   *        2
   */
  template <typename F>
  MPO<F>::MPO(const UInt dimW, const UInt dimH, const UInt length) : _dimW{dimW}, _dimH{dimH}, _length{length}, dim(4) {
    W = std::vector<Tensor::Tensor<F>>(_length);

    W[0] = Tensor::Tensor<F>({1, dimW, dimH, dimH});
    for (unsigned int l = 1; l < _length - 1; l++)
      W[l] = Tensor::Tensor<F>({dimW, dimW, dimH, dimH});
    W[_length - 1] = Tensor::Tensor<F>({dimW, 1, dimH, dimH});
  }

  template <typename F>
  MPO<F>::MPO(const Configuration::Configuration<F> &conf, const std::map<std::string, double> &P) : P{P} {

    const auto H = conf.hamiltonian;
    const auto parser = Parser::Parser<Tensor::Tensor<F>, F>(conf.config_file, P);

    _length = conf.network.length;
    _dimH = parser.dimH;
    /*@TODO: generalize this */
    _dimW = H.nearest.size() + 2;

    W = std::vector<Tensor::Tensor<F>>(_length);

    for (unsigned int l = 0; l < _length; l++) {
      if (l == 0) {
        W[l] = Tensor::Tensor<F>({1, _dimW, _dimH, _dimH});
        // Parse single site
        if (H.single_site) {
          Tensor::Tensor<F> res = parser.parse(*H.single_site, l);
          for (unsigned int i = 0; i < _dimH; i++)
            for (unsigned int j = 0; j < _dimH; j++)
              W[l][{0, 0, i, j}] = res[{i, j}];
        }
        // Parse nearest neighbourg interaction
        for (unsigned int k = 1; k <= H.nearest.size(); k++) {
          const auto &[left, right] = H.nearest[k - 1];
          Tensor::Tensor<F> res = parser.parse(left, l);
          for (unsigned int i = 0; i < _dimH; i++)
            for (unsigned int j = 0; j < _dimH; j++)
              W[l][{0, k, i, j}] = res[{i, j}];
        }
        // Write identity
        for (unsigned int i = 0; i < _dimH; i++)
          W[l][{0, _dimW - 1, i, i}] = 1.0;
      }
      if ((0 < l) && (l < _length - 1)) {
        W[l] = Tensor::Tensor<F>({_dimW, _dimW, _dimH, _dimH});
        // Parse single site
        if (H.single_site) {
          Tensor::Tensor<F> res = parser.parse(*H.single_site, l);
          for (unsigned int i = 0; i < _dimH; i++)
            for (unsigned int j = 0; j < _dimH; j++)
              W[l][{_dimW - 1, 0, i, j}] = res[{i, j}];
        }
        // Parse nearest neighbourg interaction
        for (unsigned int k = 1; k <= H.nearest.size(); k++) {
          const auto &[left, right] = H.nearest[k - 1];
          Tensor::Tensor<F> res1 = parser.parse(left, l);
          Tensor::Tensor<F> res2 = parser.parse(right, l);
          for (unsigned int i = 0; i < _dimH; i++)
            for (unsigned int j = 0; j < _dimH; j++) {
              W[l][{_dimW - 1, k, i, j}] = res1[{i, j}];
              W[l][{k, 0, i, j}] = res2[{i, j}];
            }
        }
        // Write identity
        for (unsigned int i = 0; i < _dimH; i++) {
          W[l][{0, 0, i, i}] = 1.0;
          W[l][{_dimW - 1, _dimW - 1, i, i}] = 1.0;
        }
      }
      if (l == _length - 1) {
        W[l] = Tensor::Tensor<F>({_dimW, 1, _dimH, _dimH});
        // Parse single site
        if (H.single_site) {
          Tensor::Tensor<F> res = parser.parse(*H.single_site, l);
          for (unsigned int i = 0; i < _dimH; i++)
            for (unsigned int j = 0; j < _dimH; j++)
              W[l][{_dimW - 1, 0, i, j}] = res[{i, j}];
        }
        // Parse nearest neighbourg interaction
        for (unsigned int k = 1; k <= H.nearest.size(); k++) {
          const auto &[left, right] = H.nearest[k - 1];
          Tensor::Tensor<F> res = parser.parse(right, l);
          for (unsigned int i = 0; i < _dimH; i++)
            for (unsigned int j = 0; j < _dimH; j++)
              W[l][{k, 0, i, j}] = res[{i, j}];
        }
        // Write identity
        for (unsigned int i = 0; i < _dimH; i++)
          W[l][{0, 0, i, i}] = 1.0;
      }
    }
  }

  template <typename F>
  MPO<F> MPO<F>::operator*(const MPO<F> &rhs) const {
    assert(_length == rhs._length && _dimH == rhs._dimH);

    MPO<F> result;

    result._dimW = _dimW * rhs._dimW;
    result._dimH = _dimH;
    result._length = _length;

    result.W = std::vector<Tensor::Tensor<F>>(_length);

    for (unsigned int l = 0; l < _length; l++) {
      auto P2 = rhs.W[l];

      result.W[l]("b1,b1',b2,b2',a1,a2'") = W[l]("b1,b2,a1,a2") * P2("b1',b2',a2,a2'");
      result.W[l]("b1,b1',b2,b2',a1,a2'").merge("b1,b1'");
      result.W[l]("b1'',b2,b2',a1,a2'").merge("b2,b2'");
    }
    return result;
  }

  template <typename F>
  MPO<F> &MPO<F>::compress(const UInt &dimW, const double &tolerance) {
    // std::cout << "MPO<F>::compress(" << dimW << ", " << tolerance << ") <- dimW=" << _dimW << std::endl;

    for (unsigned int l = 0; l < _length - 1; l++) {
      Tensor::Tensor<F> T1, T2;
      unsigned int r = l + 1;
      std::cout << "l=" << l << " dim=(";
      for (const auto &d : W[l].dimension())
        std::cout << d << ", ";
      std::cout << ") r=" << r << " dim=(";
      for (const auto &d : W[r].dimension())
        std::cout << d << ", ";
      std::cout << ")" << std::endl;

      auto options = Tensor::SVDOptions{Tensor::SVDNorm::equal, dimW, tolerance};

      std::vector<double> svs;
      Tensor::Contraction<F> ct = W[l]("b1,b,a1,a1'") * W[r]("b,b2,a2,a2'");
      std::tie(T1, svs, T2) = ct.SVD({"b1,b,a1,a1'", "b,b2,a2,a2'"}, options);

      W[l] = T1;
      W[r] = T2;
    }

    return *this;
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const MPO<F> &t) {

    for (UInt i = 1; i <= t.length(); i++)
      out << "MPO[" << i << "]=" << t[i] << std::endl;

    return out;
  }
} // namespace TNT::Operator

template class TNT::Operator::MPO<double>;
template class TNT::Operator::MPO<std::complex<double>>;

template std::ostream &TNT::Operator::operator<<<double>(std::ostream &, const MPO<double> &);
template std::ostream &TNT::Operator::operator<<<std::complex<double>>(std::ostream &,
                                                                       const MPO<std::complex<double>> &);
