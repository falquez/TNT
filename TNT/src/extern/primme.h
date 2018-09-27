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

#ifndef _TNT_PRIMME_H
#define _TNT_PRIMME_H

#include <primme.h>

extern "C" void dprimme_svds_convTestFun(double *sval, void *leftsvec_, void *rightsvec_,
                                         double *rNorm, int *isConv,
                                         primme_svds_params *primme_svds, int *ierr);
extern "C" void zprimme_svds_convTestFun(double *sval, void *leftsvec_, void *rightsvec_,
                                         double *rNorm, int *isConv,
                                         primme_svds_params *primme_svds, int *ierr);

namespace TNT::PRIMME {
  template <typename F>
  int calculate_ewp_primme(double *evals, F *evecs, double *resNorms, primme_params *primme);

  template <typename F>
  int calculate_svds_primme(double *svals, F *svecs, double *resNorms,
                            primme_svds_params *primme_svds);

  template <typename F>
  void svds_convergence_test(double *sval, void *leftsvec, void *rightsvec, double *rNorm,
                             int *isConv, primme_svds_params *primme_svds, int *ierr);
} // namespace TNT::PRIMME

#endif // _TNT_PRIMME_H
