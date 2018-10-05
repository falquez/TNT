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

#include <TNT/operator/observable.h>

namespace TNT::Operator {

  template <typename F>
  Observable<F>::Observable(const std::string &name, const ObservableType &kind, const Configuration::Operator<F> &op)
      : _kind{kind}, name{name} {
    if (!op.rows.empty()) {
      _length = 1;
      _dimH = op.rows.size();
      O = std::vector<Tensor::Tensor<F>>(_length);
      O[0] = Tensor::Tensor<F>({_dimH, _dimH});
      for (unsigned int i = 0; i < _dimH; i++)
        for (unsigned int j = 0; j < _dimH; j++)
          O[0][{i, j}] = op.rows[i][j];
    }
  }

  template <typename F>
  const Tensor::Tensor<F> &Observable<F>::operator[](unsigned int site) const {
    return O[site - 1];
  }

} // namespace TNT::Operator

template class TNT::Operator::Observable<double>;
template class TNT::Operator::Observable<std::complex<double>>;
