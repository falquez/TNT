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

#include <TNT/operator/projection.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/tensor.h>

#include "../algebra/algebra.h"
#include "../parser/parser.h"

namespace TNT::Operator {

  template <typename F>
  Projection<F>::Projection(const std::string &config_file, const Configuration::MPO &mpo, const UInt length,
                            const std::map<std::string, double> &P)
      : _length{length}, P{P} {

    const auto parser = Parser::Parser<Tensor::Tensor<F>, F>(config_file, P);

    //_length = length > 0 ? length : conf.network.length;
    _dimH = parser.dimH;
    /*@TODO: generalize this */
    _dimW = mpo.nearest.size();

    W = std::vector<Tensor::Tensor<F>>(_length);

    if (_length == 2) {
      std::cout << "Initializing l=2" << std::endl;
      W[0] = Tensor::Tensor<F>({_dimW, _dimH, _dimH});
      W[1] = Tensor::Tensor<F>({_dimW, _dimH, _dimH});
      for (unsigned int k = 0; k < mpo.nearest.size(); k++) {
        const auto &[left, right] = mpo.nearest[k];
        Tensor::Tensor<F> tl = parser.parse(left, 0);
        Tensor::Tensor<F> tr = parser.parse(right, 1);
        for (unsigned int i = 0; i < _dimH; i++) {
          for (unsigned int j = 0; j < _dimH; j++) {
            W[0][{k, i, j}] = tl[{i, j}];
            W[1][{k, i, j}] = tr[{i, j}];
          }
        }
      }
    } else {

      for (unsigned int l = 0; l < _length; l++) {
        if (l == 0) {
          W[l] = Tensor::Tensor<F>({1, _dimW, _dimH, _dimH});
          // Parse single site
          /*if (mpo.single_site) {
            Tensor::Tensor<F> res = parser.parse(*mpo.single_site, l);
            for (unsigned int i = 0; i < _dimH; i++)
              for (unsigned int j = 0; j < _dimH; j++)
                W[l][{0, 0, i, j}] = res[{i, j}];
          }*/
          // Parse nearest neighbourg interaction
          for (unsigned int k = 0; k < mpo.nearest.size(); k++) {
            const auto &[left, right] = mpo.nearest[k];
            Tensor::Tensor<F> res = parser.parse(left, l);
            for (unsigned int i = 0; i < _dimH; i++)
              for (unsigned int j = 0; j < _dimH; j++)
                W[l][{0, k, i, j}] = res[{i, j}];
          }
          // Write identity
          // for (unsigned int i = 0; i < _dimH; i++)
          //  W[l][{0, _dimW - 1, i, i}] = 1.0;
        }
        if ((0 < l) && (l < _length - 1)) {
          W[l] = Tensor::Tensor<F>({_dimW, _dimW, _dimH, _dimH});
          // Parse single site
          /*if (mpo.single_site) {
            Tensor::Tensor<F> res = parser.parse(*mpo.single_site, l);
            for (unsigned int i = 0; i < _dimH; i++)
              for (unsigned int j = 0; j < _dimH; j++)
                W[l][{_dimW - 1, 0, i, j}] = res[{i, j}];
          }*/
          // Parse nearest neighbourg interaction
          for (unsigned int k1 = 0; k1 < mpo.nearest.size(); k1++) {
            for (unsigned int k2 = 0; k2 < mpo.nearest.size(); k2++) {
              const auto &[_l, r1] = mpo.nearest[k1];
              const auto &[l2, _r] = mpo.nearest[k2];
              Tensor::Tensor<F> tr1 = parser.parse(r1, l);
              Tensor::Tensor<F> tl2 = parser.parse(l2, l);
              Tensor::Tensor<F> res;
              res("a,b") = tr1("a,c") * tl2("c,b");
              for (unsigned int i = 0; i < _dimH; i++)
                for (unsigned int j = 0; j < _dimH; j++) {
                  W[l][{k1, k2, i, j}] = res[{i, j}];
                }
            }
          }
          // Write identity
          /*for (unsigned int i = 0; i < _dimH; i++) {
            W[l][{0, 0, i, i}] = 1.0;
            W[l][{_dimW - 1, _dimW - 1, i, i}] = 1.0;
          }*/
        }
        if (l == _length - 1) {
          W[l] = Tensor::Tensor<F>({_dimW, 1, _dimH, _dimH});
          // Parse single site
          /*if (mpo.single_site) {
            Tensor::Tensor<F> res = parser.parse(*mpo.single_site, l);
            for (unsigned int i = 0; i < _dimH; i++)
              for (unsigned int j = 0; j < _dimH; j++)
                W[l][{_dimW - 1, 0, i, j}] = res[{i, j}];
          }*/
          // Parse nearest neighbourg interaction
          for (unsigned int k = 0; k < mpo.nearest.size(); k++) {
            const auto &[left, right] = mpo.nearest[k];
            Tensor::Tensor<F> res = parser.parse(right, l);
            for (unsigned int i = 0; i < _dimH; i++)
              for (unsigned int j = 0; j < _dimH; j++)
                W[l][{k, 0, i, j}] = res[{i, j}];
          }
          // Write identity
          // for (unsigned int i = 0; i < _dimH; i++)
          //  W[l][{0, 0, i, i}] = 1.0;
        }
      }
    }
  }

  template <typename F>
  std::ostream &operator<<(std::ostream &out, const Projection<F> &t) {

    for (UInt i = 1; i <= t.length(); i++)
      out << "Projection[" << i << "]=" << t[i] << std::endl;

    return out;
  }

} // namespace TNT::Operator

template class TNT::Operator::Projection<double>;
template class TNT::Operator::Projection<std::complex<double>>;

template std::ostream &TNT::Operator::operator<<<double>(std::ostream &, const Projection<double> &);
template std::ostream &TNT::Operator::operator<<<std::complex<double>>(std::ostream &,
                                                                       const Projection<std::complex<double>> &);
