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

#include "algebra.h"

#include <hptt/hptt.h>

namespace TNT::Algebra {

  template <typename F>
  int transpose(const std::vector<UInt> &dim, const std::vector<UInt> &trs, F *source, F *target, double alpha,
                double beta) {
    int err = 0;

    std::vector<int> dim1(dim.begin(), dim.end());
    std::vector<int> trs1(trs.begin(), trs.end());

    auto plan = hptt::create_plan(trs1, trs1.size(), alpha, source, dim1, {}, beta, target, {}, hptt::ESTIMATE,
                                  Algebra::numThreads);
    plan->execute();

    return err;
  }

  /*template <typename F>
  int transpose(const std::vector<int> &dim, const std::vector<int> &ldA, const std::vector<int> &ldB,
                const std::vector<int> &trs, F *A, F *B) {
    int err = 0;

    auto plan = hptt::create_plan(trs, trs.size(), 1.0, A, dim, ldA, 0.0, B, ldB, hptt::ESTIMATE, Algebra::numThreads);
    plan->execute();
    return err;
  }*/
} // namespace TNT::Algebra

template int TNT::Algebra::transpose<double>(const std::vector<UInt> &dim, const std::vector<UInt> &trs, double *source,
                                             double *target, double alpha, double beta);
template int TNT::Algebra::transpose<std::complex<double>>(const std::vector<UInt> &dim, const std::vector<UInt> &trs,
                                                           std::complex<double> *source, std::complex<double> *target,
                                                           double alpha, double beta);

/*template int TNT::Algebra::transpose<double>(const std::vector<int> &dim, const std::vector<int> &ldA,
                                             const std::vector<int> &ldB, const std::vector<int> &trs, double *A,
                                             double *B);
template int TNT::Algebra::transpose<std::complex<double>>(const std::vector<int> &dim, const std::vector<int> &ldA,
                                                           const std::vector<int> &ldB, const std::vector<int> &trs,
                                                           std::complex<double> *A, std::complex<double> *B);*/
