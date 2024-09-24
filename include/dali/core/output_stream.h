// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_OUTPUT_STREAM_H_
#define DALI_CORE_OUTPUT_STREAM_H_

#include <cstdio>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include "dali/core/api_helper.h"

namespace dali {

using ssize_t = std::make_signed_t<size_t>;


/**
 * @brief An abstract file-like interface for writing data.
 */
class OutputStream {
 public:
  virtual ~OutputStream() = default;

  /**
   * @brief Writes all requested data to the stream; if not all of the data can be write,
   *        an exception is thrown.
   *
   * @param buf   the output buffer
   * @param bytes the number of bytes to write
   */
  inline void WriteBytes(const void *buf, size_t bytes) {
    const char *b = static_cast<const char *>(buf);
    while (bytes) {
      ssize_t n = Write(b, bytes);
      if (n <= 0)
        throw std::runtime_error("An error occurred while writing data.");
      b += n;
      assert(static_cast<size_t>(n) <= bytes);
      bytes -= n;
    }
  }

  /**
   * @brief Writes one object of given type to the stream
   *
   * @tparam T  the type of the object to write; should be trivially copyable or otherwise
   *            safe to be overwritten with memcpy or similar.
   */
  template <typename T>
  inline T WriteOne(const T &t) {
    WriteAll(&t, 1);
    return t;
  }

  /**
   * @brief Writes `count` instances of type T to the stream to the provided buffer
   *
   * If the function cannot write the requested number of objects, an exception is thrown
   *
   * @tparam T    the type of the object to write; should be trivially copyable or otherwise
   *              safe to be overwritten with memcpy or similar.
   * @param buf   the output buffer
   * @param count the number of objects to write
   */
  template <typename T>
  inline void WriteAll(const T *buf, size_t count) {
    WriteBytes(buf, sizeof(T) * count);
  }

  /**
   * @brief Skips `count` objects in the stream
   *
   * Skips over the length of `count` objects of given type (by default char,
   * because sizeof(char) == 1).
   *
   * NOTE: Negative skips are allowed.
   *
   * @tparam T type of the object to skip; defaults to `char`
   * @param count the number of objects to skip
   */
  template <typename T = char>
  void SkipWrite(ssize_t count = 1) {
    SeekWrite(count * sizeof(T), SEEK_CUR);
  }

  /**
   * @brief Writes data to the stream and advances the write pointer; partial writes are allowed.
   *
   * A valid implementation of this function writes up to `bytes` bytes to the stream and
   * stores them in `buf`. If the function cannot write all of the requested bytes due to
   * end-of-file, it shall write all it can and return the number of bytes actually write.
   * When writing to a regular file and the file pointer is alwritey at the end, the function
   * shall return 0.
   *
   * This function does not throw EndOfStream.
   *
   * @param buf       the output buffer
   * @param bytes     the number of bytes to write
   * @return size _t  the number of bytes actually written or
   *                  <= 0 if an error occured
   */
  virtual size_t Write(const void *buf, size_t bytes) = 0;

  /**
   * @brief Moves the write pointer in the stream.
   *
   * If the new pointer would be out of range, the function may either clamp it to a valid range
   * or throw an error.
   *
   * @param pos     the offset to move
   * @param whence  the beginning - SEEK_SET, SEEK_CUR or SEEK_END
   */
  virtual void SeekWrite(ptrdiff_t pos, int whence = SEEK_SET) = 0;

  /**
   * @brief Returns the current write position, in bytes from the beginnging, in the stream.
   */
  virtual ssize_t TellWrite() const = 0;
};


class MemOutputStream : public OutputStream {
 public:
  MemOutputStream() = default;
  ~MemOutputStream() = default;
  MemOutputStream(void *mem, size_t bytes) {
    start_ = static_cast<char *>(mem);
    size_ = bytes;
  }

  size_t Write(const void *buf, size_t bytes) override {
    ptrdiff_t left = size_ - pos_;
    if (left < static_cast<ptrdiff_t>(bytes))
      bytes = left;
    std::memcpy(start_ + pos_, buf, bytes);
    pos_ += bytes;
    return bytes;
  }

  ssize_t TellWrite() const override {
    return pos_;
  }

  void SeekWrite(ssize_t offset, int whence = SEEK_SET) override {
    if (whence == SEEK_CUR) {
      offset += pos_;
    } else if (whence == SEEK_END) {
      offset += size_;
    } else {
      assert(whence == SEEK_SET);
    }
    if (offset < 0 || offset > size_)
      throw std::out_of_range("The requested position in the stream is out of range");
    pos_ = offset;
  }

  size_t Size() const {
    return size_;
  }

  ssize_t SSize() const {
    return size_;
  }

 private:
  char *start_ = nullptr;
  ptrdiff_t size_ = 0;
  ptrdiff_t pos_ = 0;
};

class FileOutputStream : public OutputStream {
 public:
  FileOutputStream() = default;
  ~FileOutputStream() { Close(); }
  explicit FileOutputStream(FILE *file, bool own_handle = true)
  : file_(file), own_handle_(own_handle) {}

  void Close() {
    if (file_) {
      if (own_handle_)
        fclose(file_);
      file_ = nullptr;
    }
  }

  size_t Write(const void *buf, size_t bytes) override {
    if (!file_)
      throw std::logic_error("The file is not open.");
    return fwrite(buf, 1, bytes, file_);
  }

  ssize_t TellWrite() const override {
    if (!file_)
      throw std::logic_error("The file is not open.");
    return ftell(file_);
  }

  void SeekWrite(ssize_t offset, int whence = SEEK_SET) override {
    if (!file_)
      throw std::logic_error("The file is not open.");
    fseek(file_, offset, whence);
  }

 private:
  FILE *file_ = nullptr;
  bool own_handle_ = false;
};

}  // namespace dali

#endif  // DALI_CORE_OUTPUT_STREAM_H_
