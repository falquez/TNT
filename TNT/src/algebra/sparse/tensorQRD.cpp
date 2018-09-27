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

#include "../../extern/lapack.h"
#include "../../util/util.h"
#include "algebra.h"

#include <hptt/hptt.h>

namespace TNT::Algebra::Sparse {

  template <typename F>
  int tensorQRD(const std::vector<UInt> &dim, const std::array<std::vector<UInt>, 2> &idx, F *data,
                F *R) {
    int err = 0;

    std::vector<int> tr1(dim.size());
    std::vector<int> tr2(dim.size());
    std::vector<int> dimQ(dim.size());
    std::vector<int> dimQ1(dim.size());

    std::copy(std::begin(idx[0]), std::end(idx[0]), std::begin(tr1));
    std::copy(std::begin(idx[1]), std::end(idx[1]), std::begin(tr1) + idx[0].size());

    for (int i = 0; i < tr1.size(); i++)
      tr2[tr1[i]] = i;

    for (int i = 0; i < dim.size(); i++)
      dimQ1[i] = dim[i];

    for (int i = 0; i < tr1.size(); i++)
      dimQ[i] = dim[tr1[i]];

    unsigned int dimM = Util::multiply(Util::selectU(dim, idx[0]));
    unsigned int dimN = Util::multiply(Util::selectU(dim, idx[1]));

    if (dimM < dimN)
      return -1;

    std::unique_ptr<F[]> Q1 = std::make_unique<F[]>(dimM * dimN);
    std::unique_ptr<F[]> tau = std::make_unique<F[]>(dimN);

    auto planQ1 = hptt::create_plan(tr1, tr1.size(), 1.0, data, dimQ1, {}, 0.0, Q1.get(), {},
                                    hptt::ESTIMATE, Algebra::numThreads);
    planQ1->execute();

    // Find QR decomposition
    err = LAPACK::geqrf(dimM, dimN, Q1.get(), dimM, tau.get());

    for (unsigned int i = 0; i < dimN; i++) {
      for (unsigned int j = i; j < dimN; j++) {
        R[j * dimN + i] = Q1[j * dimM + i];
      }
    }
    // Write Q in-place
    err = LAPACK::ungqr(dimM, dimN, dimN, Q1.get(), dimM, tau.get());

    auto planQ2 = hptt::create_plan(tr2, tr2.size(), 1.0, Q1.get(), dimQ, {}, 0.0, data, {},
                                    hptt::ESTIMATE, Algebra::numThreads);
    planQ2->execute();

    return err;
  }
} // namespace TNT::Algebra::Sparse

template int TNT::Algebra::Sparse::tensorQRD<double>(const std::vector<UInt> &dim,
                                                     const std::array<std::vector<UInt>, 2> &idx,
                                                     double *data, double *R);
template int TNT::Algebra::Sparse::tensorQRD<std::complex<double>>(
    const std::vector<UInt> &dim, const std::array<std::vector<UInt>, 2> &idx,
    std::complex<double> *data, std::complex<double> *R);
