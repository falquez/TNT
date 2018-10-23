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

    /*unsigned int lDim = 1;
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
    }*/
  }

  template <typename F>
  MPS<F>::MPS(const Configuration::Configuration<F> &conf) {

    auto hamiltonian = conf.hamiltonian;
    auto network = conf.network;

    dimH = hamiltonian.dim;
    dimB = network.dimB;
    length = network.length;
    conv_tolerance = conf.tolerance("convergence");

    _A = std::vector<Tensor::Tensor<F>>(length);
    //_LC = std::vector<Tensor::Tensor<F>>(length + 1);
    //_RC = std::vector<Tensor::Tensor<F>>(length + 1);

    /*if (conf.restart) {
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
      }*/

    /*
    // Initialize Right Contractions
    // std::cout << "Initializing Right Contractions" << std::endl;
    _RC[L] = Tensor::Tensor<F>({1, 1, 1}, 1.0);
    for (unsigned int l = L - 1; l >= i_l; l--) {
      Tensor::Tensor DW(W[l + 1]);
      RC[l]("b1,a1,a1'") =
          A[n][l + 1]("s,a1,a2") * DW("b1,b2,s,s'") * RC[l + 1]("b2,a2,a2'") * A[n][l + 1].conjugate()("s',a1',a2'");
    }

    // Initialize Left Contractions
    // std::cout << "Initializing Left Contractions" << std::endl;
    LC[1] = Tensor::Tensor<NumericalType>({1, 1, 1}, 1.0);
    for (unsigned int l = 1; l < i_r; l++) {
      Tensor::Tensor DW(W[l]);
      LC[l + 1]("b2,a2,a2'") =
          A[n][l]("s,a1,a2") * DW("b1,b2,s,s'") * LC[l]("b1,a1,a1'") * A[n][l].conjugate()("s',a1',a2'");
    }*/

    // for(ULong l=0;l<length;l++){
    // Tensor::writeToFile(_A[l],
    // output_dir+"/MPS/"+Util::format(l,length)+".hdf5");
    //}
  }

  template <typename F>
  MPS<F> &MPS<F>::initialize() {
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
    return *this;
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
      T[next]("b2,a2,a2'") = T[curr]("b1,a1,a1'") * _A[l]("s1,a1,a2") * mpo[l + 1]("b1,b2,s1,s1'") * _Ac("s1',a1',a2'");
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
      auto W = Tensor::Tensor(mpo[l + 1].dense());
      int curr = l % 2;
      int next = (l + 1) % 2;
      Tensor::Tensor<F> _Ac = _A[l].conjugate();
      T[next]("b2,a2,a2'") = T[curr]("b1,a1,a1'") * _A[l]("s1,a1,a2") * W("b1,b2,s1,s1'") * _Ac("s1',a1',a2'");
    }

    F result = T[length % 2][0];

    return result;
  }

  // Scalar product <A|B>
  template <typename F>
  F MPS<F>::operator()(const MPS<F> &B) const {
    assert(length == B.length && dimH == B.dimH);
    std::array<Tensor::Tensor<F>, 2> T;

    T[0] = Tensor::Tensor<F>({1, 1}, 1.0);
    for (unsigned int l = 0; l < length; l++) {
      unsigned int curr = l % 2;
      unsigned int next = (l + 1) % 2;
      // Tensor::Tensor<F> _Ac = _A[l].conjugate();
      T[next]("a2,a2'") = T[curr]("a1,a1'") * B._A[l]("s1,a1,a2") * _A[l].conjugate()("s1,a1',a2'");
    }

    F result = T[length % 2][0];

    return result;
  }

  // Scalar product <A|B> without given B sites
  template <typename F>
  Tensor::Tensor<F> MPS<F>::operator()(const MPS<F> &B, const std::vector<unsigned long> &sites) const {
    assert(length == B.length && dimH == B.dimH);
    // Currently support only single site or 2-site
    assert(sites.size() == 1 || (sites.size() == 2 && sites[0] == sites[1] - 1));

    std::vector<unsigned long> ml(2);
    ml[0] = sites[0] - 1;
    ml[1] = sites.size() == 1 ? ml[0] : ml[0] + 1;

    std::array<Tensor::Tensor<F>, 2> L, R;

    // std::cout << "Projector sites " << sites.size() << ": " << ml[0] << "," << ml[1] << std::endl;

    L[0] = Tensor::Tensor<F>({1, 1}, 1.0);
    for (unsigned long l = 0; l < ml[0]; l++) {
      unsigned int curr = l % 2;
      unsigned int next = (l + 1) % 2;
      Tensor::Tensor<F> _Ac = _A[l].conjugate();
      // std::cout << "Contracting " << l << std::endl;
      L[next]("a2,a2'") = L[curr]("a1,a1'") * B._A[l]("s1,a1,a2") * _Ac("s1,a1',a2'");
    }

    // std::cout << "L[" << ml[0] % 2 << "]=" << L[ml[0] % 2] << std::endl;

    R[1] = Tensor::Tensor<F>({1, 1}, 1.0);
    for (unsigned long l = 1; l < length - ml[1]; l++) {
      unsigned int curr = l % 2;
      unsigned int next = (l + 1) % 2;
      Tensor::Tensor<F> _Ac = _A[length - l].conjugate();
      // std::cout << "Contracting " << length - l << std::endl;
      R[next]("a1,a1'") = R[curr]("a2,a2'") * B._A[length - l]("s1,a1,a2") * _Ac("s1,a1',a2'");
    }
    // std::cout << "R[" << (length - ml[1]) % 2 << "]=" << R[(length - ml[1]) % 2] << std::endl;

    Tensor::Tensor<F> P, _Ac1, _Ac2;
    switch (sites.size()) {
    case 1:
      _Ac1 = _A[ml[0]].conjugate();
      P("s1,a1,a2") = L[ml[0] % 2]("a1,a1'") * _Ac1("s1,a1',a2'") * R[(length - ml[1]) % 2]("a2,a2'");
      break;
    case 2:
      _Ac1 = _A[ml[0]].conjugate();
      _Ac2 = _A[ml[1]].conjugate();
      // std::cout << "Contracting " << ml[0] << ", " << ml[1] << std::endl;
      // std::cout << "_Ac1[" << ml[0] << "]" << _Ac1("s1,a1',a") << std::endl;
      // std::cout << "_Ac2[" << ml[1] << "]" << _Ac2("s2,a,a2'") << std::endl;
      P("s1,s2,a1,a2") =
          L[ml[0] % 2]("a1,a1'") * _Ac1("s1,a1',a") * _Ac2("s2,a,a2'") * R[(length - ml[1]) % 2]("a2,a2'");
      // std::cout << "P(\"s1,s2,a1,a2\")" << P("s1,s2,a1,a2") << std::endl;
      break;
    default:
      throw std::invalid_argument("Number of sites > 2 not yet supported");
      break;
    }

    return P;
  }

  template <typename F>
  std::vector<Measurement<F>> MPS<F>::operator()(const Operator::Observable<F> &O) const {
    std::vector<Measurement<F>> result;

    switch (O.kind()) {
    case Operator::ObservableType::Site:
      // std::cout << "Measuring site observable " << O.name << " O = " << O[1] << std::endl;
      for (unsigned int l1 = 0; l1 < length; l1++) {

        std::array<Tensor::Tensor<F>, 2> T;
        T[0] = Tensor::Tensor<F>({1, 1}, 1.0);

        for (unsigned int l = 0; l < length; l++) {
          unsigned int curr = l % 2;
          unsigned int next = (l + 1) % 2;
          Tensor::Tensor<F> _Ac = _A[l].conjugate();
          if (l == l1) {
            T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * O[1]("s1,s2") * _Ac("s2,a1',a2'");
          } else {
            T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * _Ac("s1,a1',a2'");
          }
        }
        result.push_back({std::vector<ULong>{l1}, T[length % 2][0]});
      }
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
            unsigned int curr = l % 2;
            unsigned int next = (l + 1) % 2;
            Tensor::Tensor<F> _Ac = _A[l].conjugate();

            if (l == l1 && l == l2) {
              T[next]("a2,a2'") =
                  T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * Obs[0]("s1,s'") * Obs[1]("s',s2") * _Ac("s2,a1',a2'");
            } else if (l == l1 || l == l2) {
              T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * Obs[0]("s1,s2") * _Ac("s2,a1',a2'");
            } else {
              T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * _Ac("s1,a1',a2'");
            }
          }
          result.push_back({std::vector<ULong>{l1, l2}, T[length % 2][0]});
        }
      }
      break;
    case Operator::ObservableType::Shift: {
      unsigned int s = O.shift;
      Tensor::Tensor<F> R;
      std::array<Tensor::Tensor<F>, 2> T;

      T[1]("a1,a1',a2,a2'") = _A[0]("s,a1,a2") * _A[s].conjugate()("s,a1',a2'");
      for (unsigned int l = 1; l < length; l++) {
        unsigned int curr = l % 2;
        unsigned int next = (l + 1) % 2;
        T[next]("a1,a1',a3,a3'") =
            T[curr]("a1,a1',a2,a2'") * _A[l]("s,a2,a3") * _A[(l + s) % length].conjugate()("s,a2',a3'");
      }
      R = T[length % 2]("a1,a1',a2,a2'").trace("a1',a2'");
      // R("") = T[0]("a1,a1',a2,a2'") * T[1]("a1,a1',a3,a3'");
      result.push_back({std::vector<ULong>{s}, R[0]});
    } break;
    default:
      break;
    }
    return result;
  }

  /*template <typename F>
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
            T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * O("s1,s2") * _Ac("s2,a1',a2'");
          } else {
            T[next]("a2,a2'") = T[curr]("a1,a1'") * _A[l]("s1,a1,a2") * _Ac("s1,a1',a2'");
          }
        }
        res[{l1, l2}] = T[length % 2][0];
      }
    }

    return res;
  }*/

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
