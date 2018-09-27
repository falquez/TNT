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

#ifndef _TNT_BLAS_H
#define _TNT_BLAS_H

#include <array>
#include <cblas.h>
#include <complex>

namespace TNT::BLAS {
  template <typename F>
  F dot(const int &n, const std::array<F *, 2> data);

  template <>
  double dot(const int &n, const std::array<double *, 2> data) {
    return cblas_ddot(n, data[0], 1, data[1], 1);
  }

  template <>
  std::complex<double> dot(const int &n, const std::array<std::complex<double> *, 2> data) {
    return cblas_zdotu(n, data[0], 1, data[1], 1);
  }
} // namespace TNT::BLAS

#endif //_TNT_BLAS_H
