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

#include <malloc.h>

#include "../../extern/primme.h"
#include "../../util/util.h"
#include "algebra.h"

template <typename F>
void SparseTensorVec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
                     primme_params *primme, int *ierr) {

  TNT::Algebra::Sparse::SparseTensorData<F> *mdata =
      static_cast<TNT::Algebra::Sparse::SparseTensorData<F> *>(primme->matrix);

  *ierr = 0;
  for (int i = 0; i < *blockSize; i++) {

    TNT::Tensor::Sparse::Tensor<F> X(mdata->dim[0]);
    TNT::Tensor::Sparse::Tensor<F> Y; //(mdata->dim[1]);

    X.readFrom((F *)x + (*ldx) * i);

    const TNT::Tensor::Sparse::Tensor<F> *p1 = &((*mdata->vec)(mdata->subM));
    const TNT::Tensor::Sparse::Tensor<F> *p2 = &(X(mdata->sub[0]));

    Y = TNT::Tensor::Sparse::contract2<F>(mdata->sub[1], mdata->dim[1], {p1, p2});

    Y.writeTo((F *)y + (*ldy) * i);
  }
}

namespace TNT::Algebra::Sparse {

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub,
                  const Tensor::Sparse::Tensor<F> *vec, const Options &options) {
    int err = 0;
    /* Not Implemented */
    return -1;
  }

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub,
                  const Tensor::Sparse::Contraction<F> &seq, const Options &options) {
    int err = 0;
    Tensor::Sparse::Tensor<F> T;
    std::unique_ptr<double[]> targetShifts;

    /* Allocate space for converged Ritz values and residual norms */
    std::unique_ptr<double[]> rnorm = std::make_unique<double[]>(options.nv);
    std::array<std::vector<std::string>, 2> idx;
    std::array<std::vector<UInt>, 2> dim;

    std::string tsub = sub[0] + "," + sub[1];
    T(tsub) = seq;
    std::cout << "T.dataSize()=" << T.dataSize() << " T.size()=" << T.size() << std::endl;
    T.purge();
    std::cout << "T.dataSize()=" << T.dataSize() << " T.size()=" << T.size() << std::endl;

    malloc_trim(0);

    double normT = T.norm2();

    for (unsigned int n = 0; n < 2; n++) {
      idx[n] = Util::split(sub[n], ",");
      dim[n] = std::vector<UInt>(idx[n].size());
      for (unsigned int i = 0; i < dim[n].size(); i++)
        dim[n][i] = seq.dim_map.at(idx[n][i]);
    }

    SparseTensorData<F> mdata{&T, dim, sub, tsub};

    primme_params primme;
    primme_initialize(&primme);

    primme.matrix = &mdata;
    primme.matrixMatvec = SparseTensorVec<F>;

    /* Set problem parameters */
    primme.n = Util::multiply(dim[0]);
    primme.numEvals = options.nv;
    primme.initSize = options.initial;

    /* ||r|| <= eps * ||matrix|| */
    primme.aNorm = normT;
    primme.eps = options.tolerance;

    switch (options.target) {
    case Target::smallest:
      primme.target = primme_smallest;
      break;
    case Target::closest_leq:
      targetShifts = std::make_unique<double[]>(options.targets.size());
      for (unsigned int i = 0; i < options.targets.size(); i++) {
        targetShifts[i] = options.targets[i];
      }

      primme.target = primme_closest_leq;
      primme.numTargetShifts = options.targets.size();
      primme.targetShifts = targetShifts.get();
      break;
    case Target::closest_abs:
      targetShifts = std::make_unique<double[]>(options.targets.size());
      for (unsigned int i = 0; i < options.targets.size(); i++)
        targetShifts[i] = options.targets[i];

      primme.target = primme_closest_abs;
      primme.numTargetShifts = options.targets.size();
      primme.targetShifts = targetShifts.get();
      break;
    }

    err = primme_set_method(PRIMME_DYNAMIC, &primme);
    primme.printLevel = 4;

    primme_display_params(primme);
    /* Call primme  */
    err = PRIMME::calculate_ewp_primme<F>(evals, evecs, rnorm.get(), &primme);
    int converged = primme.initSize;

    primme_free(&primme);

    malloc_trim(0);

    return converged;
  }
} // namespace TNT::Algebra::Sparse

template int TNT::Algebra::Sparse::tensorEigen<double>(
    double *evals, double *evecs, const std::array<std::string, 2> &sub,
    const Tensor::Sparse::Contraction<double> &seq, const Options &options);
template int TNT::Algebra::Sparse::tensorEigen<std::complex<double>>(
    double *evals, std::complex<double> *evecs, const std::array<std::string, 2> &sub,
    const Tensor::Sparse::Contraction<std::complex<double>> &seq, const Options &options);