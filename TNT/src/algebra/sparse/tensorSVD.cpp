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

#include "../../extern/primme.h"
#include "../../util/util.h"
#include "algebra.h"

template <typename F>
void SparseTensorVec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, int *tr,
                     primme_svds_params *primme_svds, int *ierr) {

  TNT::Algebra::Sparse::SparseTensorData2<F> *matrix =
      static_cast<TNT::Algebra::Sparse::SparseTensorData2<F> *>(primme_svds->matrix);

  *ierr = 0;
  for (int i = 0; i < *blockSize; i++) {
    TNT::Tensor::Sparse::Tensor<F> X(matrix->dim[1]);
    TNT::Tensor::Sparse::Tensor<F> Y(matrix->dim[2]);
    if (*tr == 0) {
      X.readFrom((F *)x + (*ldx) * i);
      const TNT::Tensor::Sparse::Tensor<F> *p1 = &((*matrix->vec)(matrix->subscript[0]));
      const TNT::Tensor::Sparse::Tensor<F> *p2 = &(X(matrix->subscript[1]));
      Y = TNT::Tensor::Sparse::contract2<F>(matrix->subscript[2], matrix->dim[2], {p1, p2});
      Y.writeTo((F *)y + (*ldy) * i);
    } else {
      Y.readFrom((F *)x + (*ldx) * i);
      const TNT::Tensor::Sparse::Tensor<F> *p1 = &((*matrix->vec)(matrix->subscript[0]));
      const TNT::Tensor::Sparse::Tensor<F> *p2 = &(Y(matrix->subscript[2]));
      X = TNT::Tensor::Sparse::contract2<F>(matrix->subscript[1], matrix->dim[1], {p1, p2});
      X.writeTo((F *)y + (*ldy) * i);
    }
  }
}

namespace TNT::Algebra::Sparse {

  template <typename F>
  int tensorSVD(const Tensor::Sparse::Tensor<F> &vec, const std::array<std::vector<UInt>, 2> &idx,
                double *svals, F *svecs, const Options &options) {
    int err = 0;

    std::unique_ptr<double[]> rnorm;
    std::unique_ptr<uint8_t[]> workspace1;
    std::unique_ptr<uint8_t[]> workspace2;

    std::vector<uint> dim = vec.dimension();
    std::vector<uint> dimA(dim.begin(), dim.end());
    std::vector<int> idxA(dim.size());
    for (uint i = 0; i < dim.size(); i++)
      idxA[i] = i;
    std::array<std::vector<uint>, 2> dims;
    std::array<std::vector<int>, 2> idxs;
    for (int n = 0; n < 2; n++)
      dims[n] = Util::selectU(dimA, idx[n]);
    idxs[0] = std::vector<int>(idx[0].begin(), idx[0].end());
    idxs[1] = std::vector<int>(idx[1].begin(), idx[1].end());

    std::array<std::vector<uint>, 3> dim_t{dimA, dims[1], dims[0]};
    std::array<std::vector<int>, 3> idx_t{idxA, idxs[1], idxs[0]};
    SparseTensorData2<F> mdata{&vec, dim_t, idx_t};

    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);

    primme_svds.matrix = &mdata;
    primme_svds.matrixMatvec = SparseTensorVec<F>;

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

    // Allocate memory
    err = PRIMME::calculate_svds_primme<F>(nullptr, nullptr, nullptr, &primme_svds);
    /* Allocate workspace memory */
    workspace1 = std::make_unique<uint8_t[]>(static_cast<std::size_t>(primme_svds.intWorkSize));
    workspace2 = std::make_unique<uint8_t[]>(primme_svds.realWorkSize);
    /* Allocate space for converged Ritz values and residual norms */
    rnorm = std::make_unique<double[]>(options.nv);

    primme_svds.intWork = reinterpret_cast<int *>(workspace1.get());
    primme_svds.realWork = reinterpret_cast<void *>(workspace2.get());

    primme_svds_display_params(primme_svds);
    primme_svds.printLevel = 4;
    err = PRIMME::calculate_svds_primme<F>(svals, svecs, rnorm.get(), &primme_svds);

    if (err != 0) {
      std::cout << "Error: " << err << "\n";
      exit(1);
    }

    int nvecs = primme_svds.initSize;
    return nvecs;
  }
} // namespace TNT::Algebra::Sparse

template int TNT::Algebra::Sparse::tensorSVD<double>(const Tensor::Sparse::Tensor<double> &vec,
                                                     const std::array<std::vector<UInt>, 2> &idx,
                                                     double *svals, double *svecs,
                                                     const Options &options);
template int TNT::Algebra::Sparse::tensorSVD<std::complex<double>>(
    const Tensor::Sparse::Tensor<std::complex<double>> &vec,
    const std::array<std::vector<UInt>, 2> &idx, double *svals, std::complex<double> *svecs,
    const Options &options);
