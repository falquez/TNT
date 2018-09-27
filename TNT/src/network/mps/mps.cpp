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
#include <numeric>
#include <string>

#include <TNT/network/mps/mps.h>
#include <TNT/tensor/contraction.h>
#include <TNT/tensor/sparse/contraction.h>

#include "../../util/util.h"

namespace TNT::Network::MPS {

  template <typename F>
  MPS<F>::MPS(const unsigned int &dimH, const ULong &length, const unsigned int &dimB)
      : dimH{dimH}, dimB{dimB}, length{length}, conv_tolerance{1E-6} {

    _A = std::vector<Tensor::Tensor<F>>(length);

    unsigned int lDim = 1;
    for (ULong l = 0; l < length / 2; l++) {
      unsigned int rDim = std::min(lDim * dimH, dimB);
      _A[l] = Tensor::Tensor<F>({dimH, lDim, rDim});
      lDim = rDim;
    }
    unsigned int rDim = 1;
    for (ULong l = length - 1; l >= length / 2; l--) {
      unsigned int lDim = std::min(rDim * dimH, dimB);
      _A[l] = Tensor::Tensor<F>({dimH, lDim, rDim});
      rDim = lDim;
    }
    for (ULong l = 0; l < _A.size(); l++) {
      _A[l].initialize();
      _A[l].normalize_QRD(1);
    }
  }

  template <typename F>
  MPS<F>::MPS(const Configuration::Configuration<F> &conf) {

    auto hamiltonian = conf.hamiltonian;
    auto network = conf.network;

    dimH = hamiltonian.dim;
    dimB = network.dimB;
    length = network.length;
    conv_tolerance = conf.tolerance("convergence") * length;

    _A = std::vector<Tensor::Tensor<F>>(length);

    if (conf.restart) {
      throw std::invalid_argument("Restart not implemented");
      // for(unsigned int l=0;l<length;l++){
      //_A[l] =
      // Tensor::Tensor<F>(output_dir+"/MPS/"+Util::format(l,length)+".hdf5",
      //"/Tensor");
      //}
    } else {
      unsigned int lDim = 1;
      for (ULong l = 0; l < length / 2; l++) {
        unsigned int rDim = std::min(lDim * dimH, dimB);
        _A[l] = Tensor::Tensor<F>({dimH, lDim, rDim});
        lDim = rDim;
      }
      unsigned int rDim = 1;
      for (ULong l = length - 1; l >= length / 2; l--) {
        unsigned int lDim = std::min(rDim * dimH, dimB);
        _A[l] = Tensor::Tensor<F>({dimH, lDim, rDim});
        rDim = lDim;
      }
      for (ULong l = 0; l < _A.size(); l++) {
        _A[l].initialize(2);
        _A[l].normalize_QRD(1);
      }

      // for(ULong l=0;l<length;l++){
      // Tensor::writeToFile(_A[l],
      // output_dir+"/MPS/"+Util::format(l,length)+".hdf5");
      //}
    }
  }

  template <typename F>
  F MPS<F>::operator()(const Operator::MPO<F> &mpo) const {
    assert(length == mpo.length() && dimH == mpo.dimH());

    std::array<Tensor::Tensor<F>, 2> T;

    T[0] = Tensor::Tensor<F>({1, 1, 1}, 1.0);
    for (unsigned int l = 0; l < length; l++) {
      int curr = l % 2;
      int next = (l + 1) % 2;
      Tensor::Tensor<F> _Ac = _A[l].conjugate();
      T[next]("b2,a2,a2'") = T[curr]("b1,a1,a1'") * _A[l]("s1,a1,a2") * mpo[l + 1]("b1,b2,s1,s1'") *
                             _Ac("s1',a1',a2'");
    }

    F result = T[length % 2][0];

    return result;
  }

  template <typename F>
  F MPS<F>::operator()(const Operator::Sparse::MPO<F> &mpo) const {
    assert(length == mpo.length() && dimH == mpo.dimH());

    std::array<Tensor::Tensor<F>, 2> T;

    T[0] = Tensor::Tensor<F>({1, 1, 1}, 1.0);
    for (unsigned int l = 0; l < length; l++) {
      auto W = Tensor::Tensor(mpo[l + 1]);
      int curr = l % 2;
      int next = (l + 1) % 2;
      Tensor::Tensor<F> _Ac = _A[l].conjugate();
      T[next]("b2,a2,a2'") =
          T[curr]("b1,a1,a1'") * _A[l]("s1,a1,a2") * W("b1,b2,s1,s1'") * _Ac("s1',a1',a2'");
    }

    F result = T[length % 2][0];

    return result;
  }

  template <typename F>
  std::vector<Measurement<F>> MPS<F>::operator()(const Operator::Observable<F> &O) const {
    std::vector<Measurement<F>> result;

    switch (O.type()) {
    case Operator::ObservableType::Site:
      break;
    case Operator::ObservableType::Correlation:

      for (unsigned int l1 = 0; l1 < length; l1++) {
        for (unsigned int l2 = l1; l2 < length; l2++) {
          std::array<Tensor::Tensor<F>, 2> Obs;
          Obs[0] = O[1];
          Obs[1] = O[1];
          std::array<Tensor::Tensor<F>, 2> T;
          T[0] = Tensor::Tensor<F>({1, 1}, 1.0);

          for (unsigned int l = 0; l < length; l++) {
            int curr = l % 2;
            int next = (l + 1) % 2;
            Tensor::Tensor<F> _Ac = _A[l].conjugate();

            if (l == l1 && l == l2) {
              T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * Obs[0]("s1,s'") *
                                  Obs[1]("s',s2") * _Ac("s2,a1',a2'");
            } else if (l == l1 || l == l2) {
              T[next]("a2,a2'") =
                  T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * Obs[0]("s1,s2") * _Ac("s2,a1',a2'");
            } else {
              T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * _Ac("s1,a1',a2'");
            }
          }
          result.push_back({std::vector<ULong>{l1, l2}, T[length % 2][0]});
        }
      }
      break;
    default:
      break;
    }
    return result;
  }

  template <typename F>
  std::map<std::array<ULong, 2>, F> MPS<F>::correlation(const Tensor::Tensor<F> &O) const {
    std::map<std::array<ULong, 2>, F> res;
    for (unsigned int l1 = 0; l1 < length; l1++) {
      for (unsigned int l2 = l1; l2 < length; l2++) {

        std::array<Tensor::Tensor<F>, 2> T;
        T[0] = Tensor::Tensor<F>({1, 1}, 1.0);

        for (unsigned int l = 0; l < length; l++) {
          int curr = l % 2;
          int next = (l + 1) % 2;
          Tensor::Tensor<F> _Ac = _A[l].conjugate();
          if (l == l1 || l == l2) {
            T[next]("a2,a2'") =
                T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * O("s1,s2") * _Ac("s2,a1',a2'");
          } else {
            T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * _Ac("s1,a1',a2'");
          }
        }
        res[{l1, l2}] = T[length % 2][0];
      }
    }

    return res;
  }

  template <typename F>
  std::tuple<ULong, ULong, Sweep::Direction> MPS<F>::position(const State &state) const {
    ULong i1 = state.iteration - 1;
    ULong l1 = length - 1;
    ULong p1 = i1 % l1;
    ULong p2 = p1 + 1;
    Sweep::Direction dir = (i1 / l1) % 2 ? Sweep::Direction::Left : Sweep::Direction::Right;
    if (dir == Sweep::Direction::Left) {
      auto tmp = p2;
      p2 = l1 - p1;
      p1 = l1 - tmp;
    }
    return {p1 + 1, p2 + 1, dir};
  }

  template <typename F>
  Iterator MPS<F>::sweep(State &state) {
    return Iterator(state, length, length * 100, conv_tolerance);
  }

} // namespace TNT::Network::MPS

template class TNT::Network::MPS::MPS<double>;
template class TNT::Network::MPS::MPS<std::complex<double>>;
