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

#ifndef _TNT_CONFIGURATION_ITERATOR_H
#define _TNT_CONFIGURATION_ITERATOR_H

#include <vector>

namespace TNT::Configuration {
  namespace Sweep {

    class Token {};

    template <typename O>
    class Iterator {
      const std::vector<O> &P;
      unsigned int current;

    public:
      explicit Iterator(const std::vector<O> &P) : P{P}, current{0} {}

      // Produce current parameter map
      std::tuple<unsigned int, O> operator*() const { return {current + 1, P[current]}; };

      Iterator &operator++() {
        ++current;

        return *this;
      }

      bool operator!=(const Token) const { return current < P.size(); }
    };
  } // namespace Sweep

  template <typename O>
  class Iterator {
    const std::vector<O> &P;

  public:
    Iterator(const std::vector<O> &P) : P{P} {}
    Sweep::Iterator<O> begin() const { return Sweep::Iterator<O>{P}; }
    Sweep::Token end() const { return {}; }
  };
} // namespace TNT::Configuration

#endif
