// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {
namespace filesystem {
vector<std::string> traverse_directories(const std::string& path, const std::string& filter);
std::string join_path(const std::string &dir, const std::string &path);
}  // namespace filesystem

struct ImageFileWrapper {
  Tensor<CPUBackend> image;
  std::string filename;
  // some field for auxiliary info to pass to the reader
  std::string meta;
};

class FileLoader : public Loader< CPUBackend, ImageFileWrapper > {
 public:
  explicit inline FileLoader(
    const OpSpec& spec,
    bool shuffle_after_epoch = false)
    : Loader<CPUBackend, ImageFileWrapper >(spec),
      file_filter_(spec.GetArgument<string>("file_filter")),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0) {

    vector<string> files;

    has_files_arg_ = spec.TryGetRepeatedArgument(files, "files");
    has_file_list_arg_ = spec.TryGetArgument(file_list_, "file_list");
    has_file_root_arg_ = spec.TryGetArgument(file_root_, "file_root");

    DALI_ENFORCE(has_file_root_arg_ || has_files_arg_ || has_file_list_arg_,
      "``file_root`` argument is required when not using ``files`` or ``file_list``.");

    DALI_ENFORCE(has_files_arg_ + has_file_list_arg_ <= 1,
      "File paths can be provided through ``files`` or ``file_list`` but not both.");

    if (has_file_list_arg_) {
      DALI_ENFORCE(!file_list_.empty(), "``file_list`` argument cannot be empty");
      if (!has_file_root_arg_) {
#ifdef WINVER
        constexpr char dir_sep = '\\';
#else
        constexpr char dir_sep = '/';
#endif
        auto idx = file_list_.rfind(dir_sep);
        if (idx != string::npos) {
          file_root_ = file_list_.substr(0, idx);
        }
      }
    }

    if (has_files_arg_) {
      DALI_ENFORCE(files.size() > 0, "``files`` specified an empty list.");
      images_ = std::move(files);
    }

    /*
    * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
    * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
    * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLoader so all
    * DALI instances will do shuffling after each epoch
    */
    if (shuffle_after_epoch_ || stick_to_shard_)
      DALI_ENFORCE(
        !shuffle_after_epoch_ || !stick_to_shard_,
        "shuffle_after_epoch and stick_to_shard cannot be both true");
    if (shuffle_after_epoch_ || shuffle_)
      DALI_ENFORCE(
        !shuffle_after_epoch_ || !shuffle_,
        "shuffle_after_epoch and random_shuffle cannot be both true");
    /*
      * Imply `stick_to_shard` from  `shuffle_after_epoch`
      */
    if (shuffle_after_epoch_) {
      stick_to_shard_ = true;
    }

    if (!dont_use_mmap_) {
      mmap_reserver = FileStream::FileStreamMappinReserver(
                                  static_cast<unsigned int>(initial_buffer_fill_));
    }
    copy_read_data_ = dont_use_mmap_ || !mmap_reserver.CanShareMappedData();
  }

  void PrepareEmpty(ImageFileWrapper &tensor) override;
  void ReadSample(ImageFileWrapper& tensor) override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    if (images_.empty()) {
      if (!has_files_arg_ && !has_file_list_arg_) {
        images_ = filesystem::traverse_directories(file_root_, file_filter_);
      } else if (has_file_list_arg_) {
        // load paths from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        string image_file;
        while (s >> image_file) {
          images_.push_back(image_file);
        }
        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
      }
    }
    DALI_ENFORCE(Size() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(images_.begin(), images_.end(), g);
    }
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, Size());
    } else {
      current_index_ = 0;
    }

    current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(images_.begin(), images_.end(), g);
    }
  }

  using Loader<CPUBackend, ImageFileWrapper >::shard_id_;
  using Loader<CPUBackend, ImageFileWrapper >::num_shards_;

  string file_list_, file_root_, file_filter_;
  vector<std::string> images_;

  bool has_files_arg_     = false;
  bool has_file_list_arg_ = false;
  bool has_file_root_arg_ = false;

  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
  FileStream::FileStreamMappinReserver mmap_reserver;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
