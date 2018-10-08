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

#include <iostream>
#include <numeric>

#include <TNT/network/mps/sweep.h>

namespace TNT::Network::MPS::Sweep {

  // Produce series
  // (1,2),(2,3),..,(L-2,L-1),(L-1,L),(L-1,L),(L-2,L-1),...(2,3),(1,2),(1,2),...
  std::tuple<ULong, ULong, Direction> Iterator::operator*() const {
    // std::cout << "Iterator::operator* iteration i=" << state.iteration << std::endl;
    ULong i1 = state.iteration - 1;
    ULong l1 = length - 1;
    ULong p1 = i1 % l1;
    ULong p2 = p1 + 1;
    Direction dir = (i1 / l1) % 2 ? Direction::Left : Direction::Right;
    if (dir == Direction::Left) {
      auto tmp = p2;
      p2 = l1 - p1;
      p1 = l1 - tmp;
    }
    return {p1 + 1, p2 + 1, dir};
  }

  Iterator &Iterator::operator++() {
    // std::cout << "Iterator::operator++ iteration i=" << state.iteration << std::endl;
    state.writeToFile();

    ++state.iteration;

    return *this;
  }

  bool Iterator::operator!=(const Token) const {
    // std::cout << "Iterator::operator!= iteration i=" << state.iteration << std::endl;
    /*ULong i1 = state.iteration - 2;
    ULong l1 = length - 1;
    ULong p1 = i1 % l1;
    ULong p2 = p1 + 1;
    Sweep::Direction dir = (i1 / l1) % 2 ? Sweep::Direction::Left : Sweep::Direction::Right;

    if (dir == Sweep::Direction::Left) {
      auto tmp = p2;
      p2 = l1 - p1;
      p1 = l1 - tmp;
    }*/

    double deltaEV = state.variance;

    bool converged = false;
    if ((std::abs(deltaEV) < conv_tolerance) && (state.iteration > 2 * length)) {
      converged = true;
    }

    return state.iteration < maxIter && !converged;
  }
} // namespace TNT::Network::MPS::Sweep
