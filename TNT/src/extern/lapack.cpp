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

#include "lapack.h"

#include <algorithm>
#include <memory>
namespace TNT::LAPACK {

  template <>
  lapack_int geqrf<double>(lapack_int m, lapack_int n, double *a, lapack_int lda, double *tau, int matrix_layout) {
    return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
  }

  template <>
  lapack_int geqrf<std::complex<double>>(lapack_int m, lapack_int n, std::complex<double> *a, lapack_int lda,
                                         std::complex<double> *tau, int matrix_layout) {
    return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);
  }

  template <>
  lapack_int gelqf<double>(lapack_int m, lapack_int n, double *a, lapack_int lda, double *tau, int matrix_layout) {
    return LAPACKE_dgelqf(matrix_layout, m, n, a, lda, tau);
  }

  template <>
  lapack_int gelqf<std::complex<double>>(lapack_int m, lapack_int n, std::complex<double> *a, lapack_int lda,
                                         std::complex<double> *tau, int matrix_layout) {
    return LAPACKE_zgelqf(matrix_layout, m, n, a, lda, tau);
  }

  template <>
  lapack_int ungqr<double>(lapack_int m, lapack_int n, lapack_int k, double *a, lapack_int lda, double *tau,
                           int matrix_layout) {
    return LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);
  }

  template <>
  lapack_int ungqr<std::complex<double>>(lapack_int m, lapack_int n, lapack_int k, std::complex<double> *a,
                                         lapack_int lda, std::complex<double> *tau, int matrix_layout) {
    return LAPACKE_zungqr(matrix_layout, m, n, k, a, lda, tau);
  }

  template <>
  lapack_int unglq<double>(lapack_int m, lapack_int n, lapack_int k, double *a, lapack_int lda, double *tau,
                           int matrix_layout) {
    return LAPACKE_dorglq(matrix_layout, m, n, k, a, lda, tau);
  }

  template <>
  lapack_int unglq<std::complex<double>>(lapack_int m, lapack_int n, lapack_int k, std::complex<double> *a,
                                         lapack_int lda, std::complex<double> *tau, int matrix_layout) {
    return LAPACKE_zunglq(matrix_layout, m, n, k, a, lda, tau);
  }

  template <>
  lapack_int gesdd<double>(char jobz, lapack_int m, lapack_int n, double *a, lapack_int lda, double *s, double *u,
                           lapack_int ldu, double *vt, lapack_int ldvt, int matrix_layout) {
    return LAPACKE_dgesdd(matrix_layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
  }

  template <>
  lapack_int gesdd<std::complex<double>>(char jobz, lapack_int m, lapack_int n, std::complex<double> *a, lapack_int lda,
                                         double *s, std::complex<double> *u, lapack_int ldu, std::complex<double> *vt,
                                         lapack_int ldvt, int matrix_layout) {
    return LAPACKE_zgesdd(matrix_layout, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
  }
  template <>
  lapack_int gesvd<double>(char jobu, char jobvt, lapack_int m, lapack_int n, double *a, lapack_int lda, double *s,
                           double *u, lapack_int ldu, double *vt, lapack_int ldvt, int matrix_layout) {
    // double *superb;

    // sizeof(superb)=MIN(m,n)-1
    auto superb = std::make_unique<double[]>(std::min(m, n) - 1);
    return LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb.get());
  }

  template <>
  lapack_int gesvd<std::complex<double>>(char jobu, char jobvt, lapack_int m, lapack_int n, std::complex<double> *a,
                                         lapack_int lda, double *s, std::complex<double> *u, lapack_int ldu,
                                         std::complex<double> *vt, lapack_int ldvt, int matrix_layout) {

    auto superb = std::make_unique<double[]>(std::min(m, n) - 1);

    return LAPACKE_zgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb.get());
  }

  template <>
  lapack_int gesvdx<double>(char jobu, char jobvt, char range, lapack_int m, lapack_int n, double *a, lapack_int lda,
                            double vl, double vu, lapack_int il, lapack_int iu, lapack_int *ns, double *s, double *u,
                            lapack_int ldu, double *vt, lapack_int ldvt, int matrix_layout) {

    auto superb = std::make_unique<lapack_int[]>(12 * std::min(m, n) - 1);

    return LAPACKE_dgesvdx(matrix_layout, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt,
                           superb.get());
  }

  template <>
  lapack_int gesvdx<std::complex<double>>(char jobu, char jobvt, char range, lapack_int m, lapack_int n,
                                          std::complex<double> *a, lapack_int lda, double vl, double vu, lapack_int il,
                                          lapack_int iu, lapack_int *ns, double *s, std::complex<double> *u,
                                          lapack_int ldu, std::complex<double> *vt, lapack_int ldvt,
                                          int matrix_layout) {
    // sizeof(superb)=12*MIN(m,n)-1
    auto superb = std::make_unique<lapack_int[]>(12 * std::min(m, n) - 1);
    return LAPACKE_zgesvdx(matrix_layout, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt,
                           superb.get());
  }

} // namespace TNT::LAPACK
