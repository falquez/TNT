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
void SparseTensorVec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme,
                     int *ierr) {

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

template <typename F>
void SparseTensorVecPX(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme,
                       int *ierr) {

  TNT::Algebra::Sparse::SparseTensorData<F> *mdata =
      static_cast<TNT::Algebra::Sparse::SparseTensorData<F> *>(primme->matrix);

  unsigned int nprojs = mdata->P.size();
  unsigned int nconsr = mdata->N.size();
  // std::cout << "nprojs=" << nprojs << std::endl;
  *ierr = 0;
  for (int i = 0; i < *blockSize; i++) {

    TNT::Tensor::Sparse::Tensor<F> X(mdata->dim[0]);
    TNT::Tensor::Sparse::Tensor<F> Y;
    X.readFrom((F *)x + (*ldx) * i);

    const TNT::Tensor::Sparse::Tensor<F> *p1 = &((*mdata->vec)(mdata->subM));
    const TNT::Tensor::Sparse::Tensor<F> *p2 = &(X(mdata->sub[0]));

    Y = TNT::Tensor::Sparse::contract2<F>(mdata->sub[1], mdata->dim[1], {p1, p2});
    Y.writeTo((F *)y + (*ldy) * i);

    for (unsigned int n = 0; n < nprojs; n++) {
      TNT::Tensor::Tensor<F> Xd(mdata->dim[0]);
      Xd.readFrom((F *)x + (*ldx) * i);
      // std::cout << "mdata->P[" << n << "]" << mdata->P[n] << std::endl;
      // std::cout << "Xd=" << Xd << std::endl;
      F c = std::get<0>(mdata->P[n])(mdata->sub[0]) * Xd(mdata->sub[0]);
      std::get<0>(mdata->P[n]).conjugate().addTo((F *)y + (*ldy) * i, std::get<1>(mdata->P[n]) * c);
    }

    for (unsigned int c_i = 0; c_i < nconsr; c_i++) {
      // TNT::Tensor::Sparse::Tensor<F> Xs(mdata->dim[0]);
      TNT::Tensor::Sparse::Tensor<F> Ys;
      // Xs.readFrom((F *)x + (*ldx) * i);
      // TNT::Tensor::Sparse::Tensor<F> Y;

      const TNT::Tensor::Sparse::Tensor<F> *p1 = &(std::get<0>(mdata->N[c_i])(mdata->subM));
      const TNT::Tensor::Sparse::Tensor<F> *p2 = &(X(mdata->sub[0]));

      Ys = TNT::Tensor::Sparse::contract2<F>(mdata->sub[1], mdata->dim[1], {p1, p2});
      Ys.purge();
      // std::cout << "mdata->P[" << n << "]" << mdata->P[n] << std::endl;
      // std::cout << "Xd=" << Xd << std::endl;
      // Ys(mdata->sub[1]) = std::get<0>(mdata->N[c_i])(mdata->subM) * Xs(mdata->sub[0]);
      TNT::Tensor::Tensor<F> Xd(X);
      TNT::Tensor::Tensor<F> Yd(Ys);
      F n = Yd(mdata->sub[0]) * Xd(mdata->sub[0]);

      // F d = n - std::get<1>(mdata->N[c_i]);
      F d = std::get<1>(mdata->N[c_i]) - std::get<2>(mdata->N[c_i]);

      // TNT::Tensor::Tensor<F> Yd2(Y);
      std::cout << "Iter Charge = " << n << " mu=" << std::get<1>(mdata->N[c_i]) << std::endl;
      Ys.addTo((F *)y + (*ldy) * i, std::get<1>(mdata->N[c_i]));

      // std::get<0>(mdata->P[n]).conjugate().addTo((F *)y + (*ldy) * i, std::get<1>(mdata->P[n]) * c);
    }
  }
}

namespace TNT::Algebra::Sparse {

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub, const Tensor::Sparse::Tensor<F> *vec,
                  const Options &options) {
    int err = 0;
    /* Not Implemented */
    return -1;
  }

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub,
                  const Tensor::Sparse::Contraction<F> &seq, const std::vector<TNT::Tensor::TensorScalar<F>> &P,
                  const std::vector<TNT::Tensor::Sparse::TensorConstraint<F>> &X, const Options &options) {
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

    SparseTensorData<F> mdata{&T, dim, sub, tsub, P, X};

    primme_params primme;
    primme_initialize(&primme);

    primme.matrix = &mdata;

    if (P.size() > 0 || X.size() > 0)
      primme.matrixMatvec = SparseTensorVecPX<F>;
    else
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
    double *, double *, const std::array<std::string, 2> &, const Tensor::Sparse::Contraction<double> &,
    const std::vector<TNT::Tensor::TensorScalar<double>> &,
    const std::vector<TNT::Tensor::Sparse::TensorConstraint<double>> &, const Options &);
template int TNT::Algebra::Sparse::tensorEigen<std::complex<double>>(
    double *, std::complex<double> *, const std::array<std::string, 2> &,
    const Tensor::Sparse::Contraction<std::complex<double>> &,
    const std::vector<TNT::Tensor::TensorScalar<std::complex<double>>> &,
    const std::vector<TNT::Tensor::Sparse::TensorConstraint<std::complex<double>>> &, const Options &);
