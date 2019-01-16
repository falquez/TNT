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

#ifndef _TNT_LAPACK_H
#define _TNT_LAPACK_H

//#include <complex>
#define lapack_complex_double std::complex<double>

#include <lapacke.h>

namespace TNT::LAPACK {
  template <typename F>
  lapack_int geqrf(lapack_int m, lapack_int n, F *a, lapack_int lda, F *tau, int matrix_layout = LAPACK_COL_MAJOR);

  template <typename F>
  lapack_int gelqf(lapack_int m, lapack_int n, F *a, lapack_int lda, F *tau, int matrix_layout = LAPACK_COL_MAJOR);

  template <typename F>
  lapack_int ungqr(lapack_int m, lapack_int n, lapack_int k, F *a, lapack_int lda, F *tau,
                   int matrix_layout = LAPACK_COL_MAJOR);

  template <typename F>
  lapack_int unglq(lapack_int m, lapack_int n, lapack_int k, F *a, lapack_int lda, F *tau,
                   int matrix_layout = LAPACK_COL_MAJOR);

  template <typename F>
  lapack_int gesdd(char jobz, lapack_int m, lapack_int n, F *a, lapack_int lda, double *s, F *u, lapack_int ldu, F *vt,
                   lapack_int ldvt, int matrix_layout = LAPACK_COL_MAJOR);

  template <typename F>
  lapack_int gesvd(char jobu, char jobvt, lapack_int m, lapack_int n, F *a, lapack_int lda, double *s, F *u,
                   lapack_int ldu, F *vt, lapack_int ldvt, int matrix_layout = LAPACK_COL_MAJOR);

  template <typename F>
  lapack_int gesvdx(char jobu, char jobvt, char range, lapack_int m, lapack_int n, F *a, lapack_int lda, double vl,
                    double vu, lapack_int il, lapack_int iu, lapack_int *ns, double *s, F *u, lapack_int ldu, F *vt,
                    lapack_int ldvt, int matrix_layout = LAPACK_COL_MAJOR);

} // namespace TNT::LAPACK

#endif //_TNT_LAPACK_H
