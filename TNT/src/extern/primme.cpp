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

#include <complex>
#include <iostream>

#include "primme.h"

namespace TNT::PRIMME {

  template <>
  int calculate_ewp_primme<double>(double *evals, double *evecs, double *resNorms,
                                   primme_params *primme) {
    int res = dprimme(evals, evecs, resNorms, primme);
    return res;
  }

  template <>
  int calculate_ewp_primme<std::complex<double>>(double *evals, std::complex<double> *evecs,
                                                 double *resNorms, primme_params *primme) {
    int res = zprimme(evals, evecs, resNorms, primme);
    return res;
  }

  template <>
  int calculate_svds_primme<double>(double *svals, double *svecs, double *resNorms,
                                    primme_svds_params *primme_svds) {
    int res = dprimme_svds(svals, svecs, resNorms, primme_svds);
    return res;
  }

  template <>
  int calculate_svds_primme<std::complex<double>>(double *svals, std::complex<double> *svecs,
                                                  double *resNorms,
                                                  primme_svds_params *primme_svds) {
    int res = zprimme_svds(svals, svecs, resNorms, primme_svds);
    return res;
  }
} // namespace TNT::PRIMME
