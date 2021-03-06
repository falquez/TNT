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

#ifndef _TNT_NETWORK_MPS_H
#define _TNT_NETWORK_MPS_H

#include <algorithm>
#include <vector>

#include <TNT/configuration/configuration.h>
#include <TNT/network/mps/sweep.h>
#include <TNT/network/network.h>
#include <TNT/operator/mpo.h>
#include <TNT/operator/observable.h>
#include <TNT/operator/projection.h>
#include <TNT/operator/sparse/mpo.h>
#include <TNT/tensor/tensor.h>

namespace TNT::Network::MPS {

  template <typename F>
  struct Measurement {
    std::vector<ULong> site;
    F value;
  };

  template <typename F>
  class MPS {
    unsigned int dimH;
    unsigned int dimB;
    ULong length;
    std::vector<Tensor::Tensor<F>> _A;
    std::vector<std::vector<double>> _S;
    std::vector<unsigned int> _mbd;

    std::string sub;

    double conv_tolerance;
    std::string config_file;

  public:
    MPS() : length{0} {}
    MPS(const Configuration::Configuration<F> &conf);

    MPS(const ULong &length, const unsigned int dimH = 1, const unsigned int dimB = 1,
        const double conv_tolerance = 1E-6);

    unsigned int size() const { return _A.size(); }

    MPS<F> &initialize();
    MPS<F> &initialize(const Operator::Projection<F> &prj);

    MPS<F> &operator()(const std::string &idx) {
      sub = idx;
      return *this;
    }

    Tensor::Tensor<F> &operator[](unsigned int site) { return _A[site - 1]; }
    std::vector<double> &SV(unsigned int site) { return _S[site - 1]; }

    unsigned int mbd(unsigned int site) const { return _mbd[site]; }
    // Norm
    F norm() const;

    // Scalar product <A|B>
    F operator()(const MPS<F> &B) const;

    // Scalar product <A|B> without given B sites
    Tensor::Tensor<F> operator()(const MPS<F> &B, const std::vector<unsigned long> &sites) const;

    F operator()(const Operator::MPO<F> &mpo) const;
    F operator()(const Operator::Sparse::MPO<F> &mpo) const;

    std::vector<Measurement<F>> operator()(const Operator::Observable<F> &O) const;
    // std::map<std::array<ULong, 2>, F> correlation(const Tensor::Tensor<F> &O) const;

    MPS<F> operator()(const Operator::Projection<F> &prj) const;

    Iterator sweep(State &state);

    std::tuple<ULong, ULong, Sweep::Direction> position(const State &state) const;
  };
} // namespace TNT::Network::MPS

#endif // _TNT_NETWORK_MPS_H
