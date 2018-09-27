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

#ifndef _TNT_NETWORK_MPS_SWEEP_H
#define _TNT_NETWORK_MPS_SWEEP_H

#include <tuple>

#include <TNT/network/network.h>

namespace TNT::Network::MPS {

  namespace Sweep {
    enum class Direction { Left, Right };

    class Token {};

    class Iterator {
      State &state;
      // ULong &iteration;
      ULong length;
      ULong maxIter;
      double conv_tolerance;
      Direction direction;

    public:
      explicit Iterator(State &state, const ULong length, const ULong maxIter,
                        double conv_tolerance)
          : state{state}, length{length}, maxIter{maxIter}, conv_tolerance{conv_tolerance} {
        direction = Direction::Right;
      }

      // Produce series
      // (1,2),(2,3),..,(L-2,L-1),(L-1,L),(L-1,L),(L-2,L-1),...(2,3),(1,2),(1,2),...
      std::tuple<ULong, ULong, Direction> operator*() const;

      Iterator &operator++();

      bool operator!=(const Token) const;
    };
  } // namespace Sweep

  class Iterator {
    State &state;

    ULong length;
    ULong maxIter;
    double conv_tolerance;

  public:
    Iterator(State &state, const ULong length, const ULong maxIter, double conv_tolerance)
        : state{state}, length{length}, maxIter{maxIter}, conv_tolerance{conv_tolerance} {}
    Sweep::Iterator begin() const {
      return Sweep::Iterator{state, length, maxIter, conv_tolerance};
    }
    Sweep::Token end() const { return {}; }
  };
} // namespace TNT::Network::MPS

#endif // _TNT_NETWORK_MPS_SWEEP_H
