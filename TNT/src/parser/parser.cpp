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

#include "parser.h"

#include <TNT/configuration/operator.h>
#include <TNT/tensor/sparse/tensor.h>
#include <TNT/tensor/tensor.h>

#include <METL/metl.h>

namespace TNT::Parser {

  template <typename T, typename F>
  Parser<T, F>::Parser(const std::string &config_file, const std::map<std::string, double> &P) : P{P} {
    throw std::invalid_argument("Not Implemented");
  };

  template <typename T, typename F>
  T Parser<T, F>::parse(const std::string &s, int pos) const {
    throw std::invalid_argument("Not Implemented");
  };

  template <>
  Parser<Tensor::Tensor<double>, double>::Parser(const std::string &config_file, const std::map<std::string, double> &P)
      : P{P} {
    // unsigned int dimH = conf.hamiltonian.dim;

    const auto operators = TNT::Configuration::Operators<double>(config_file);

    for (const auto &[name, op] : operators) {
      if (!op.rows.empty()) {
        unsigned int nrows = op.rows.size();
        unsigned int ncols = op.rows[0].size();
        Tensor::Tensor<double> o({nrows, ncols});
        for (unsigned int i = 0; i < nrows; i++)
          for (unsigned int j = 0; j < ncols; j++)
            o[{i, j}] = op.rows[i][j];
        O.emplace(name, o);
      } else if (op.sparse) {
	std::cout << "Reading sparse OP " << op.name << " " << op.size << " " << op.sparse << std::endl;
	if (op.size > 1) {
	  for (unsigned int i = 0; i < op.size; i++) {
	    Tensor::Sparse::Tensor<double> o(op.file, op.path, i);
	    O.emplace(name + std::to_string(i + 1), Tensor::Tensor<double>(o));
	  }
	} else {
	  Tensor::Sparse::Tensor<double> o(op.file, op.path);
	  O.emplace(name, Tensor::Tensor<double>(o));
	}
      } else {
        if (op.size > 1) {
          for (unsigned int i = 0; i < op.size; i++) {
            Tensor::Tensor<double> o(op.file, op.path, i);
	    O.emplace(name + std::to_string(i + 1), o);
          }
        } else {
          Tensor::Tensor<double> o(op.file, op.path);
          O.emplace(name, o);
        }
      }
    }
    /* @TODO: better strategy to read Hilbert space dimension*/
    dimH = O.begin()->second.dimension().front();
    Tensor::Tensor<double> Id({dimH, dimH});
    for (unsigned int i = 0; i < dimH; i++)
      Id[{i, i}] = 1.0;
    O.emplace("Id", Id);

    for (const auto &[name, op] : O) {
      std::cout << "Read Operator " << name << "=" << op << std::endl;
    }
  };

  template <>
  Parser<Tensor::Sparse::Tensor<double>, double>::Parser(const std::string &config_file,
							 const std::map<std::string, double> &P)
      : P{P} {

    // unsigned int dimH = conf.hamiltonian.dim;

    const auto operators = TNT::Configuration::Operators<double>(config_file);

    for (const auto &[name, op] : operators) {
      if (!op.rows.empty()) {
        unsigned int nrows = op.rows.size();
        unsigned int ncols = op.rows[0].size();
        Tensor::Sparse::Tensor<double> o({nrows, ncols});
        for (unsigned int i = 0; i < nrows; i++)
          for (unsigned int j = 0; j < ncols; j++)
            o <<= {{i, j}, op.rows[i][j]};
        O.emplace(name, o);
      } else {
        if (op.size > 1) {
          for (unsigned int i = 0; i < op.size; i++) {
            Tensor::Sparse::Tensor<double> o(op.file, op.path, i);
            O.emplace(name + std::to_string(i + 1), o);
          }
        } else {
          Tensor::Sparse::Tensor<double> o(op.file, op.path);
          O.emplace(name, o);
        }
      }
    }
    /* @TODO: better strategy to read Hilbert space dimension*/
    dimH = O.begin()->second.dimension().front();
    Tensor::Sparse::Tensor<double> Id({dimH, dimH});
    for (unsigned int i = 0; i < dimH; i++)
      Id <<= {{i, i}, 1.0};
    O.emplace("Id", Id);
  };

  template <>
  Tensor::Tensor<double> Parser<Tensor::Tensor<double>, double>::parse(const std::string &s, int pos) const {
    using F = double;

    std::cout << "Parsing expression: " << s << std::endl;
    auto compiler = metl::makeCompiler<int, F, Tensor::Tensor<F>>();
    compiler.setOperatorPrecedence("*", 5);
    compiler.setOperatorPrecedence("+", 6);
    compiler.setOperatorPrecedence("-", 6);
    compiler.setUnaryOperatorPrecedence("-", 3);
    compiler.setOperator<Tensor::Tensor<F>, Tensor::Tensor<F>>("+", [](auto l, auto r) { return l + r; });
    compiler.setOperator<Tensor::Tensor<F>, Tensor::Tensor<F>>("-", [](auto l, auto r) { return l - r; });
    compiler.setOperator<Tensor::Tensor<F>, F>("*", [](auto l, auto r) { return l * r; });
    compiler.setOperator<F, Tensor::Tensor<F>>("*", [](auto l, auto r) { return r * l; });
    compiler.setOperator<Tensor::Tensor<F>, int>("*", [](auto l, auto r) { return l * r; });
    compiler.setOperator<Tensor::Tensor<F>, int>("*", [](auto l, auto r) { return l * static_cast<F>(r); });
    compiler.setOperator<int, Tensor::Tensor<F>>("*", [](auto l, auto r) { return r * static_cast<F>(l); });
    compiler.setOperator<F, F>("*", [](auto l, auto r) { return l * r; });
    compiler.setFunction<F>("sqrt", [](auto d) { return std::sqrt(d); });
    compiler.setFunction<int>("sgn", [](auto d) { return (d % 2) ? 1 : -1; });
    compiler.setUnaryOperator<F>("-", [](auto i) { return -i; });
    compiler.setUnaryOperator<Tensor::Tensor<F>>("-", [](auto i) { return -i; });

    for (const auto &[name, value] : P) {
      compiler.setConstant(name, value);
    }
    compiler.setConstant("l", pos);
    for (const auto &[name, op] : O) {
      compiler.setConstant(name, op);
    }

    auto res = compiler.build<Tensor::Tensor<F>>(s)();

    return res;
  }

  template <>
  Tensor::Sparse::Tensor<double> Parser<Tensor::Sparse::Tensor<double>, double>::parse(const std::string &s,
										       int pos) const {
    using F = double;
    std::cout << "Parsing expression: " << s << std::endl;

    auto compiler = metl::makeCompiler<int, F, Tensor::Sparse::Tensor<F>>();
    compiler.setOperatorPrecedence("*", 5);
    compiler.setOperatorPrecedence("+", 6);
    compiler.setOperatorPrecedence("-", 6);
    compiler.setUnaryOperatorPrecedence("-", 3);
    compiler.setOperator<Tensor::Sparse::Tensor<F>, Tensor::Sparse::Tensor<F>>("+",
									       [](auto l, auto r) { return l + r; });
    compiler.setOperator<Tensor::Sparse::Tensor<F>, Tensor::Sparse::Tensor<F>>("-",
									       [](auto l, auto r) { return l - r; });
    compiler.setOperator<Tensor::Sparse::Tensor<F>, F>("*", [](auto l, auto r) { return l * r; });
    compiler.setOperator<F, Tensor::Sparse::Tensor<F>>("*", [](auto l, auto r) { return r * l; });
    compiler.setOperator<Tensor::Sparse::Tensor<F>, int>("*", [](auto l, auto r) { return l * static_cast<F>(r); });
    compiler.setOperator<int, Tensor::Sparse::Tensor<F>>("*", [](auto l, auto r) { return r * static_cast<F>(l); });

    compiler.setOperator<F, F>("*", [](auto l, auto r) { return l * r; });
    compiler.setFunction<F>("sqrt", [](auto d) { return std::sqrt(d); });
    compiler.setFunction<int>("sgn", [](auto d) { return (d % 2) ? 1 : -1; });
    compiler.setUnaryOperator<F>("-", [](auto i) { return -i; });
    compiler.setUnaryOperator<Tensor::Sparse::Tensor<F>>("-", [](auto i) { return -i; });

    for (const auto &[name, value] : P) {
      compiler.setConstant(name, value);
    }
    compiler.setConstant("l", pos);
    for (const auto &[name, op] : O) {
      compiler.setConstant(name, op);
    }

    auto res = compiler.build<Tensor::Sparse::Tensor<F>>(s)();

    std::cout << "res=" << res << std::endl;

    return res;
  }

} // namespace TNT::Parser

template class TNT::Parser::Parser<TNT::Tensor::Tensor<double>, double>;
template class TNT::Parser::Parser<TNT::Tensor::Tensor<std::complex<double>>, std::complex<double>>;

template class TNT::Parser::Parser<TNT::Tensor::Sparse::Tensor<double>, double>;
template class TNT::Parser::Parser<TNT::Tensor::Sparse::Tensor<std::complex<double>>, std::complex<double>>;
