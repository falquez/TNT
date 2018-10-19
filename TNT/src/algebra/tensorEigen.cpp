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

#include "../extern/primme.h"
#include "../util/util.h"
#include "algebra.h"

/*template <typename F>
struct TensorData {
  const TNT::Tensor::Contraction<F> &seq;
  std::vector<unsigned int> dimX;
  std::string subX;
  const std::vector<TNT::Tensor::TensorScalar<F>> &P;
  const std::vector<TNT::Tensor::TensorScalar<F>> &X;
};*/

template <typename F>
struct TensorData {
  const TNT::Tensor::Contraction<F> &seq;
  std::array<std::vector<TNT::Algebra::UInt>, 2> dim;
  std::array<std::string, 2> sub;
  // std::string subM;
  const std::vector<TNT::Tensor::TensorScalar<F>> &P;
  const std::vector<TNT::Tensor::Sparse::TensorConstraint<F>> &N;
};

template <typename F>
void tensorVec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {

  TensorData<F> *mdata = static_cast<TensorData<F> *>(primme->matrix);

  TNT::Tensor::Contraction<F> seq{mdata->seq}; //{*(TNT::Tensor::Contraction<F> *)primme->matrix};
  // seq.data.push_back(nullptr);

  *ierr = 0;
  for (int i = 0; i < *blockSize; i++) {
    seq.data.back() = (F *)x + (*ldx) * i;
    *ierr = TNT::Algebra::tensorMult<F>((F *)y + (*ldy) * i, seq.subs.back(), seq);
  }
}

template <typename F>
void tensorVecPX(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {

  TensorData<F> *mdata = static_cast<TensorData<F> *>(primme->matrix);

  TNT::Tensor::Contraction<F> seq{mdata->seq}; //{*(TNT::Tensor::Contraction<F> *)primme->matrix};
  seq.data.push_back(nullptr);

  unsigned int nprojs = mdata->P.size();
  // std::cout << "nprojs=" << nprojs << " dimX=";
  // for (const auto &d : mdata->dimX)
  //  std::cout << d << ",";
  // std::cout << " subX=" << mdata->subX << std::endl;

  *ierr = 0;
  for (int i = 0; i < *blockSize; i++) {
    seq.data.back() = (F *)x + (*ldx) * i;
    *ierr = TNT::Algebra::tensorMult<F>((F *)y + (*ldy) * i, seq.subs.back(), seq);

    for (unsigned int n = 0; n < nprojs; n++) {
      TNT::Tensor::Tensor<F> X(mdata->dimX);
      X.readFrom((F *)x + (*ldx) * i);
      // std::cout << "mdata->P[" << n << "]" << mdata->P[n] << std::endl;
      // std::cout << "X=" << X << std::endl;
      F c = std::get<0>(mdata->P[n])(mdata->subX) * X(mdata->subX);
      // std::cout << "n=" << n << " c=" << c << std::endl;
      std::get<0>(mdata->P[n]).conjugate().addTo((F *)y + (*ldy) * i, std::get<1>(mdata->P[n]) * c);
    }
  }
}

namespace TNT::Algebra {
  // const size_t alignment = 64;

  template <typename F>
  int tensorEigen(double *evals, F *evecs, const std::array<std::string, 2> &sub, const Tensor::Contraction<F> &seq,
                  const std::vector<TNT::Tensor::TensorScalar<F>> &P,
                  const std::vector<TNT::Tensor::TensorScalar<F>> &X, const Options &options) {
    int err = 0;

    std::unique_ptr<double[]> targetShifts;

    /* Allocate space for converged Ritz values and residual norms */
    std::unique_ptr<double[]> rnorm = std::make_unique<double[]>(options.nv);

    Tensor::Contraction<F> nseq{seq};

    auto idx = Util::split(sub[0], ",");
    std::vector<UInt> dim(idx.size());
    for (unsigned int i = 0; i < dim.size(); i++)
      dim[i] = seq.dim_map.at(idx[i]);

    nseq.dims.push_back(dim);
    nseq.subs.push_back(sub[0]);
    nseq.subs.push_back(sub[1]);

    primme_params primme;
    primme_initialize(&primme);

    TensorData<F> mdata{nseq, dim, sub[0], P, X};

    primme.matrix = &mdata;
    if (P.size() > 0 || X.size() > 0)
      primme.matrixMatvec = tensorVecPX<F>;
    else
      primme.matrixMatvec = tensorVec<F>;
    // primme.matrixMatvec = tensorVecSeq<F>;
    /* Set problem parameters */
    primme.n = Util::multiply(dim);
    primme.numEvals = options.nv;

    /* ||r|| <= eps * ||matrix|| */
    primme.eps = options.tolerance;
    switch (options.target) {
    case Target::largest:
      primme.target = primme_largest;
      break;
    case Target::smallest:
      primme.target = primme_smallest;
      break;
    case Target::closest_leq:
      targetShifts = std::make_unique<double[]>(options.targets.size());
      for (unsigned int i = 0; i < options.targets.size(); i++)
        targetShifts[i] = options.targets[i];

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

    /* Call primme  */
    primme.printLevel = 4;
    err = PRIMME::calculate_ewp_primme<F>(evals, evecs, rnorm.get(), &primme);
    int converged = primme.initSize;

    primme_free(&primme);

    return converged;
  }
} // namespace TNT::Algebra

template int TNT::Algebra::tensorEigen<double>(double *evals, double *evecs, const std::array<std::string, 2> &sub,
                                               const Tensor::Contraction<double> &seq,
                                               const std::vector<TNT::Tensor::TensorScalar<double>> &P = {},
                                               const std::vector<TNT::Tensor::TensorScalar<double>> &X = {},
                                               const Options &options);
template int TNT::Algebra::tensorEigen<std::complex<double>>(
    double *evals, std::complex<double> *evecs, const std::array<std::string, 2> &sub,
    const Tensor::Contraction<std::complex<double>> &seq,
    const std::vector<TNT::Tensor::TensorScalar<std::complex<double>>> &P = {},
    const std::vector<TNT::Tensor::TensorScalar<std::complex<double>>> &X = {}, const Options &options);
