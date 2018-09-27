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

#include <numeric>

#include "../extern/primme.h"
#include "../util/util.h"
#include "algebra.h"

#include <hptt/hptt.h>
#include <tcl/tcl.h>

template <typename F>
struct MatVecData {
  MatVecData(F *data, const std::array<std::vector<int>, 3> &dim,
             const std::array<std::vector<int>, 3> &idx)
      : data(data), dim(dim) {
    subscript[0] = TNT::Util::stringify(idx[0]);
    subscript[1] = TNT::Util::stringify(idx[1]);
    subscript[2] = TNT::Util::stringify(idx[2]);
  }
  F *data;
  std::array<std::vector<int>, 3> dim;
  std::array<std::string, 3> subscript;
};

template <typename F>
void MatVec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, int *tr,
            primme_svds_params *primme_svds, int *ierr) {

  tcl::error err;
  MatVecData<F> *matrix;

  matrix = (MatVecData<F> *)primme_svds->matrix;

  std::vector<int> dimA = matrix->dim[0];
  std::vector<int> dimB = matrix->dim[1];
  std::vector<int> dimC = matrix->dim[2];
  std::string subscriptA = matrix->subscript[0];
  std::string subscriptB = matrix->subscript[1];
  std::string subscriptC = matrix->subscript[2];

  tcl::Tensor<F> A(dimA, (matrix->data));

  *ierr = 0;

  for (int b = 0; b < *blockSize; b++) {
    if (*tr == 0) {
      tcl::Tensor<F> B(dimB, ((F *)x) + (*ldx) * b);
      tcl::Tensor<F> C(dimC, ((F *)y) + (*ldy) * b);
      err = tcl::tensorMult<F>(1.0, A[subscriptA], B[subscriptB], 0.0, C[subscriptC]);
    } else {
      tcl::Tensor<F> B(dimB, ((F *)y) + (*ldy) * b);
      tcl::Tensor<F> C(dimC, ((F *)x) + (*ldx) * b);
      err = tcl::tensorMult<F>(1.0, A[subscriptA], C[subscriptC], 0.0, B[subscriptB]);
    }
    if (err != 0) {
      std::cout << "TensorMult Error: " << err << " " << tcl::getErrorString(err) << "\n";
      exit(err);
    }
  }
}

namespace TNT::Algebra {

  template <typename F>
  int tensorSVD(int idx, const std::vector<int> &dim, F *data, F *side, const Options &options) {
    int err = 0;

    std::vector<int> dimA = dim;
    std::vector<int> dimB = {dim[idx]};
    std::vector<int> dimC = dim;
    dimC.erase(dimC.begin() + idx);

    std::vector<int> idxA(dim.size());
    std::iota(std::begin(idxA), std::end(idxA), 0);
    std::vector<int> idxB = {idx};
    std::vector<int> idxC(dim.size() - 1);
    auto it =
        std::copy_if(idxA.begin(), idxA.end(), idxC.begin(), [idx](int i) { return i != idx; });

    MatVecData<F> mdata{data, {dimA, dimB, dimC}, {idxA, idxB, idxC}};

    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);

    primme_svds.matrix = &mdata;
    primme_svds.matrixMatvec = MatVec<F>;

    /* Set problem parameters */
    primme_svds.m = Util::multiply(dim) / dim[idx];
    ;
    primme_svds.n = dim[idx];
    primme_svds.numSvals = options.nv;

    /* ||r|| <= eps * ||matrix|| */
    primme_svds.eps = options.tolerance;
    switch (options.target) {
    case Target::largest:
      primme_svds.target = primme_svds_largest;
      break;
    case Target::smallest:
      primme_svds.target = primme_svds_smallest;
      break;
    }
    /* Allocate space for converged Ritz values and residual norms */
    std::unique_ptr<double[]> rnorm = std::make_unique<double[]>(primme_svds.numSvals);
    std::unique_ptr<double[]> svals = std::make_unique<double[]>(primme_svds.numSvals);
    std::unique_ptr<F[]> svecs =
        std::make_unique<F[]>((primme_svds.n + primme_svds.m) * primme_svds.numSvals);

    /* Call primme_svds  */
    primme_svds.printLevel = 4;
    err = PRIMME::calculate_svds_primme<F>(svals.get(), svecs.get(), rnorm.get(), &primme_svds);
    // free memory
    primme_svds_free(&primme_svds);

    int nvecs = primme_svds.initSize;

    // Copy the left singular vectors U_i to data, correcting the permutation.
    std::vector<int> dim_tr = dimC;
    dim_tr.push_back(dim[idx]);
    std::vector<int> idx_tr(dim.size() - 1);
    std::iota(std::begin(idx_tr), std::end(idx_tr), 0);
    idx_tr.insert(idx_tr.begin(), dim.size() - 1);

    auto plan = hptt::create_plan(idx_tr, idx_tr.size(), 1.0, svecs.get(), dim_tr, {}, 0.0, data,
                                  {}, hptt::ESTIMATE, 1);
    plan->execute();

    // Copy the right singular vectors V_i to side;
    for (int i = 0; i < dim[idx]; i++) {
      for (int j = 0; j < nvecs; j++)
        side[i + j * dim[idx]] = svals[j] * svecs[nvecs * primme_svds.m + j * primme_svds.n + i];
    }

    return nvecs;
  }

  template <typename F>
  int tensorSVD(const std::vector<UInt> &dim, const std::array<std::vector<UInt>, 2> &idx,
                double *svals, F *svecs, F *data, const Options &options) {
    int err = 0;

    std::unique_ptr<double[]> rnorm;
    std::unique_ptr<uint8_t[]> workspace1;
    std::unique_ptr<uint8_t[]> workspace2;

    std::vector<int> dimA(dim.begin(), dim.end());
    std::vector<int> idxA(dim.size());
    std::array<std::vector<int>, 2> dims;
    std::array<std::vector<int>, 2> idxs;

    for (int i = 0; i < dim.size(); i++)
      idxA[i] = i;
    for (int n = 0; n < 2; n++)
      dims[n] = Util::selectU(dimA, idx[n]);

    idxs[0] = std::vector<int>(idx[0].begin(), idx[0].end());
    idxs[1] = std::vector<int>(idx[1].begin(), idx[1].end());

    std::array<std::vector<int>, 3> dim_t{dimA, dims[1], dims[0]};
    std::array<std::vector<int>, 3> idx_t{idxA, idxs[1], idxs[0]};
    MatVecData<F> mdata{data, dim_t, idx_t};

    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);

    primme_svds.matrix = &mdata;
    primme_svds.matrixMatvec = MatVec<F>;

    /* Set problem parameters */
    primme_svds.m = Util::multiply(dims[0]);
    primme_svds.n = Util::multiply(dims[1]);
    primme_svds.mLocal = primme_svds.m;
    primme_svds.nLocal = primme_svds.n;

    primme_svds.numSvals = options.nv;

    primme_svds.eps = options.tolerance;
    primme_svds.aNorm = options.aNorm;
    primme_svds.initSize = options.initial;
    switch (options.target) {
    case Target::largest:
      primme_svds.target = primme_svds_largest;
      break;
    case Target::smallest:
      primme_svds.target = primme_svds_smallest;
      break;
    }

    /* Set method to solve the singular value problem and
       the underneath eigenvalue problem (optional) */
    primme_svds_set_method(primme_svds_default, PRIMME_DEFAULT_METHOD, PRIMME_DEFAULT_METHOD,
                           &primme_svds);

    /* Call primme_svds  */
    /* Allocate space for converged Ritz values and residual norms */
    // Allocate memory
    err = PRIMME::calculate_svds_primme<F>(nullptr, nullptr, nullptr, &primme_svds);

    workspace1 = std::make_unique<uint8_t[]>(static_cast<std::size_t>(primme_svds.intWorkSize));
    workspace2 = std::make_unique<uint8_t[]>(primme_svds.realWorkSize);
    rnorm = std::make_unique<double[]>(options.nv);

    primme_svds.intWork = reinterpret_cast<int *>(workspace1.get());
    primme_svds.realWork = reinterpret_cast<void *>(workspace2.get());

    if (options.verbosity > 0) {
      primme_svds_display_params(primme_svds);
      primme_svds.printLevel = 4;
    }

    err = PRIMME::calculate_svds_primme<F>(svals, svecs, rnorm.get(), &primme_svds);

    if (err != 0) {
      std::cout << "Error: " << err << "\n";
      exit(1);
    }

    int nvecs = primme_svds.initSize;

    return nvecs;
  }
} // namespace TNT::Algebra

template int TNT::Algebra::tensorSVD<double>(const int idx, const std::vector<int> &dim,
                                             double *data, double *side, const Options &options);
template int TNT::Algebra::tensorSVD<std::complex<double>>(const int idx,
                                                           const std::vector<int> &dim,
                                                           std::complex<double> *data,
                                                           std::complex<double> *side,
                                                           const Options &options);

template int TNT::Algebra::tensorSVD<double>(const std::vector<UInt> &dim,
                                             const std::array<std::vector<UInt>, 2> &idx,
                                             double *svals, double *svecs, double *data,
                                             const Options &options);
template int TNT::Algebra::tensorSVD<std::complex<double>>(
    const std::vector<UInt> &dim, const std::array<std::vector<UInt>, 2> &idx, double *svals,
    std::complex<double> *svecs, std::complex<double> *data, const Options &options);