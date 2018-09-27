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

#ifndef _TNT_STORAGE_H
#define _TNT_STORAGE_H

#include <hdf5.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace TNT::Storage {

  namespace Data {
    template <unsigned int n, typename F>
    struct Sparse {
      uint idx[n];
      F v;
    };

    template <typename F>
    struct SparseL {
      unsigned long long idx;
      F v;
    };

    template <typename F>
    struct Dense {
      std::vector<hsize_t> dim;
      std::vector<hsize_t> max_dim;
      std::unique_ptr<F[]> data;
    };

    template <typename F>
    struct Metadata {
      std::string name;
      F value;
    };
  } // namespace Data
  namespace Type {
    template <unsigned int n, typename F>
    class Sparse {
      typedef TNT::Storage::Data::Sparse<n, F> t;
      hid_t type_id;

    public:
      Sparse() {
        hsize_t dim = n;
        hid_t array_id = H5Tarray_create(H5T_NATIVE_UINT, 1, &dim);
        type_id = H5Tcreate(H5T_COMPOUND, sizeof(t));
        H5Tinsert(type_id, "Index", HOFFSET(t, idx), array_id);
        H5Tinsert(type_id, "Value", HOFFSET(t, v), H5T_NATIVE_DOUBLE);
      }
      hid_t operator()() { return type_id; }
      ~Sparse() { H5Tclose(type_id); }
    };

    template <typename F>
    class SparseL {
      typedef TNT::Storage::Data::SparseL<F> t;
      hid_t type_id;

    public:
      SparseL() {
        type_id = H5Tcreate(H5T_COMPOUND, sizeof(t));
        H5Tinsert(type_id, "Index", HOFFSET(t, idx), H5T_NATIVE_ULLONG);
        H5Tinsert(type_id, "Value", HOFFSET(t, v), H5T_NATIVE_DOUBLE);
      }
      hid_t operator()() { return type_id; }
      ~SparseL() { H5Tclose(type_id); }
    };

  } // namespace Type

  enum class FileMode { CreateNew, CreateOverwrite, ReadOnly, ReadWrite };

  struct FileException : public std::runtime_error {
    FileException(std::string const &message) : std::runtime_error(message) {}
  };

  class Storage {
    std::string filename;
    hid_t file;

  public:
    Storage() : file{H5I_INVALID_HID} {}

    Storage(const std::string &filename, const FileMode &filemode);

    Storage &operator=(Storage &&lhs) {
      filename = lhs.filename;
      file = lhs.file;

      lhs.filename = "";
      lhs.file = H5I_INVALID_HID;

      return *this;
    }

    ~Storage() {
      if (file != H5I_INVALID_HID)
        H5Fclose(file);
    }

    template <typename O>
    int read(const std::string &name, O &obj) const;

    template <typename O>
    int write(const std::string &name, const O &obj);

    template <typename O>
    int create(const std::string &name, const O &obj);

    int create_group(const std::string &path);
  };
} // namespace TNT::Storage
#endif // _TNT_STORAGE_H
