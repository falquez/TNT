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

#include "../util/util.h"
#include <TNT/storage/storage.h>

namespace TNT::Storage {

  Storage::Storage(const std::string &filename, const FileMode &filemode) : filename{filename} {
    unsigned int flags = 0;

    switch (filemode) {
    case FileMode::CreateNew:
      file = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
      break;
    case FileMode::CreateOverwrite:
      file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      break;
    case FileMode::ReadOnly:
      if (Util::file_exists(filename))
        file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      else
        file = H5I_INVALID_HID;
      break;
    case FileMode::ReadWrite:
      if (Util::file_exists(filename))
        file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      else
        file = H5I_INVALID_HID;
      break;
    }
    if (file < 0) {
      std::cout << "File Error" << std::endl;
      throw FileException("Error opening file");
    }
  }

  int Storage::create_group(const std::string &path) {

    auto gcpl = H5Pcreate(H5P_LINK_CREATE);
    H5Pset_create_intermediate_group(gcpl, 1);
    auto group = H5Gcreate(file, path.c_str(), gcpl, H5P_DEFAULT, H5P_DEFAULT);
    H5Pclose(gcpl);
    auto status = H5Gclose(group);

    return status;
  }

  template <>
  int Storage::create<Data::Metadata<unsigned int>>(const std::string &name, const Data::Metadata<unsigned int> &meta) {
    hid_t datatype = H5T_NATIVE_UINT;

    hsize_t size = 1;

    auto group = H5Gopen(file, name.c_str(), H5P_DEFAULT);
    auto space = H5Screate_simple(1, &size, nullptr);
    auto attribute = H5Acreate(group, meta.name.c_str(), datatype, space, H5P_DEFAULT, H5P_DEFAULT);
    auto status = H5Awrite(attribute, datatype, &meta.value);

    H5Aclose(attribute);
    H5Sclose(space);
    H5Gclose(group);

    return status;
  }

  template <>
  int Storage::create<Data::Metadata<unsigned long>>(const std::string &name,
						     const Data::Metadata<unsigned long> &meta) {
    hid_t datatype = H5T_NATIVE_ULONG;

    hsize_t size = 1;

    auto group = H5Gopen(file, name.c_str(), H5P_DEFAULT);
    auto space = H5Screate_simple(1, &size, nullptr);
    auto attribute = H5Acreate(group, meta.name.c_str(), datatype, space, H5P_DEFAULT, H5P_DEFAULT);
    auto status = H5Awrite(attribute, datatype, &meta.value);

    H5Aclose(attribute);
    H5Sclose(space);
    H5Gclose(group);

    return status;
  }

  template <>
  int Storage::create<Data::Metadata<std::vector<unsigned int>>>(
      const std::string &name, const Data::Metadata<std::vector<unsigned int>> &meta) {
    std::vector<hsize_t> dims(meta.value.size(), 1);
    hsize_t dim = dims.size();

    auto datatype = H5Tarray_create(H5T_NATIVE_UINT, 1, &dim);

    hsize_t size = 1;

    auto group = H5Gopen(file, name.c_str(), H5P_DEFAULT);
    auto space = H5Screate_simple(1, &size, nullptr);
    auto attribute = H5Acreate(group, meta.name.c_str(), datatype, space, H5P_DEFAULT, H5P_DEFAULT);
    auto status = H5Awrite(attribute, datatype, &meta.value[0]);

    H5Tclose(datatype);
    H5Aclose(attribute);
    H5Sclose(space);
    H5Gclose(group);

    return status;
  }

  template <>
  int Storage::create<Data::Dense<double>>(const std::string &name, const Data::Dense<double> &dense) {

    auto datatype = H5T_NATIVE_DOUBLE;

    auto space = H5Screate_simple(dense.dim.size(), &dense.dim[0], nullptr);
    auto dataset = H5Dcreate(file, name.c_str(), datatype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto status = H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dense.data.get());

    H5Dclose(dataset);
    H5Sclose(space);

    return status;
  }

  template <>
  int Storage::create<std::vector<Data::SparseL<double>>>(const std::string &name,
							  const std::vector<Data::SparseL<double>> &obj) {

    hsize_t size = obj.size();
    auto datatype = Type::SparseL<double>();
    auto space = H5Screate_simple(1, &size, nullptr);

    auto dataset = H5Dcreate(file, name.c_str(), datatype(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int status = H5Dwrite(dataset, datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &obj[0]);

    H5Dclose(dataset);
    H5Sclose(space);
    return status;
  }

  template <>
  int Storage::create<std::vector<Data::Sparse<2, double>>>(const std::string &name,
							    const std::vector<Data::Sparse<2, double>> &obj) {

    hsize_t size = obj.size();

    auto datatype = Type::Sparse<2, double>();
    auto space = H5Screate_simple(1, &size, nullptr);

    auto dataset = H5Dcreate(file, name.c_str(), datatype(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int status = H5Dwrite(dataset, datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &obj[0]);

    H5Dclose(dataset);
    H5Sclose(space);

    return status;
  }

  template <>
  int Storage::create<Data::Dense<std::complex<double>>>(const std::string &name,
							 const Data::Dense<std::complex<double>> &data) {

    throw std::domain_error("Storage::create<Data::Dense<std::complex<double>>> Not implemented");
  }

  template <>
  int Storage::write<Data::Dense<std::complex<double>>>(const std::string &name,
							const Data::Dense<std::complex<double>> &data) {

    throw std::domain_error("Storage::write<Data::Dense<std::complex<double>>> Not implemented");
  }

  template <>
  int Storage::write<std::vector<Data::Sparse<2, double>>>(const std::string &name,
							   const std::vector<Data::Sparse<2, double>> &obj) {

    hsize_t size = obj.size();

    auto datatype = Type::Sparse<2, double>();
    auto space = H5Screate_simple(1, &size, nullptr);

    auto dataset = H5Dcreate(file, name.c_str(), datatype(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int status = H5Dwrite(dataset, datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &obj[0]);

    H5Dclose(dataset);
    H5Sclose(space);

    return status;
  }

  template <>
  int Storage::write<std::vector<Data::SparseL<double>>>(const std::string &name,
							 const std::vector<Data::SparseL<double>> &obj) {
    throw std::domain_error("Storage::write<std::vector<Data::SparseL<double>>> Not implemented");
  }

  template <>
  int Storage::write<std::vector<Data::SparseL<std::complex<double>>>>(
      const std::string &name, const std::vector<Data::SparseL<std::complex<double>>> &data) {

    throw std::domain_error("Storage::write<Data::SparseL<std::complex<double>>>> Not implemented");
  }

  template <>
  int Storage::create<std::vector<Data::SparseL<std::complex<double>>>>(
      const std::string &name, const std::vector<Data::SparseL<std::complex<double>>> &data) {

    throw std::domain_error("Storage::write<Data::SparseL<std::complex<double>>>> Not implemented");
  }

  template <>
  int Storage::write<Data::Dense<double>>(const std::string &name, const Data::Dense<double> &dense) {

    auto datatype = H5T_NATIVE_DOUBLE;
    auto dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
    auto status = H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dense.data.get());

    H5Dclose(dataset);

    return status;
  }

  template <>
  int Storage::read<Data::Dense<double>>(const std::string &name, Data::Dense<double> &dense) const {

    auto datatype = H5T_NATIVE_DOUBLE;
    auto dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);

    auto space = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(space);

    dense.dim.resize(rank);
    dense.max_dim.resize(rank);

    H5Sget_simple_extent_dims(space, &dense.dim[0], &dense.max_dim[0]);

    unsigned int totalSize = 1;
    for (int i = 0; i < dense.dim.size(); i++)
      totalSize *= dense.dim[i];
    dense.data = std::make_unique<double[]>(totalSize);

    auto status = H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, dense.data.get());

    H5Sclose(space);
    H5Dclose(dataset);

    return status;
  }

  template <>
  int Storage::read<Data::Dense<std::complex<double>>>(const std::string &name,
						       Data::Dense<std::complex<double>> &dense) const {
    std::cout << "ERROR: Storage::read<Data::Dense<std::complex<double>>> not "
                 "implemented";
    exit(1);
  }

  template <>
  int Storage::read<std::vector<Data::Sparse<2, double>>>(const std::string &name,
							  std::vector<Data::Sparse<2, double>> &obj) const {

    auto datatype = Type::Sparse<2, double>();
    auto dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);

    auto space = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(rank);
    std::vector<hsize_t> max_dims(rank);
    H5Sget_simple_extent_dims(space, &dims[0], &max_dims[0]);

    hsize_t size = 1;
    for (const auto &d : dims)
      size = d * size;

    std::vector<Data::Sparse<2, double>> data(size);

    auto status = H5Dread(dataset, datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);

    H5Sclose(space);
    H5Dclose(dataset);

    obj = data;
    return status;
  }

  template <>
  int Storage::read<std::vector<Data::SparseL<double>>>(const std::string &name,
							std::vector<Data::SparseL<double>> &obj) const {

    auto datatype = Type::SparseL<double>();
    auto dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);

    auto space = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(rank);
    std::vector<hsize_t> max_dims(rank);
    H5Sget_simple_extent_dims(space, &dims[0], &max_dims[0]);

    hsize_t size = 1;
    for (const auto &d : dims)
      size = d * size;

    std::vector<Data::SparseL<double>> data(size);

    auto status = H5Dread(dataset, datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);

    H5Sclose(space);
    H5Dclose(dataset);

    obj = data;
    return status;
  }

  template <>
  int Storage::read<std::vector<Data::SparseL<std::complex<double>>>>(
      const std::string &name, std::vector<Data::SparseL<std::complex<double>>> &obj) const {
    throw std::domain_error("Storage::read<Data::SparseL<std::complex<double>>>> Not implemented");
  }

  template <>
  int Storage::write<Data::Metadata<unsigned int>>(const std::string &name, const Data::Metadata<unsigned int> &meta) {

    auto group = H5Gopen(file, name.c_str(), H5P_DEFAULT);

    hsize_t size = 1;
    auto space = H5Screate_simple(1, &size, nullptr);
    auto attribute = H5Acreate(group, meta.name.c_str(), H5T_NATIVE_UINT, space, H5P_DEFAULT, H5P_DEFAULT);
    auto status = H5Awrite(attribute, H5T_NATIVE_UINT, &meta.value);

    H5Aclose(attribute);
    H5Sclose(space);
    H5Gclose(group);

    return status;
  }

  template <>
  int Storage::read<Data::Metadata<unsigned int>>(const std::string &name, Data::Metadata<unsigned int> &meta) const {
    auto group = H5Gopen(file, name.c_str(), H5P_DEFAULT);
    auto attribute = H5Aopen(group, meta.name.c_str(), H5P_DEFAULT);
    auto status = H5Aread(attribute, H5T_NATIVE_UINT, &meta.value);

    H5Aclose(attribute);
    H5Gclose(group);

    return status;
  }

  template <>
  int Storage::read<Data::Metadata<std::vector<unsigned int>>>(const std::string &name,
							       Data::Metadata<std::vector<unsigned int>> &meta) const {

    auto group = H5Gopen(file, name.c_str(), H5P_DEFAULT);
    auto attribute = H5Aopen(group, meta.name.c_str(), H5P_DEFAULT);
    auto datatype = H5Aget_type(attribute);

    hsize_t dim;
    int ndim = H5Tget_array_dims(datatype, &dim);

    meta.value.resize(dim);
    auto status = H5Aread(attribute, datatype, &meta.value[0]);

    H5Tclose(datatype);
    H5Aclose(attribute);
    H5Gclose(group);

    return status;
  }

} // namespace TNT::Storage
