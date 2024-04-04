// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_

#include <stdexcept>
#include <list>
#include <memory>

namespace dali {
/**
 * CachingList differs from std::List by the ability to recycle empty elements. When allocating
 * memory is expensive it is better to store already allocated but no longer needed element in the
 * list of the free elements, than to free the memory and allocate it again later. CachingList
 * supports the following operations:
 * - GetEmpty  moves an empty element of type T, either allocate it or use one from the free list
 * - PopFront  moves the element from the front and removes it from the full list;
 *             an exception is thrown if the list is empty
 * - PeekFront returns a reference to the element at the front; if the list is empty;
 *             an exception is thrown if the list is empty
 * - Recycle   moves passed element to the free list
 * - PushBack  moves element to the full list
 * - IsEmpty   checks if the full list is empty
 * All functions operate on one element list as transferring elements between list is a very low
 * cost operation, which doesn't involve any memory allocation, while adding an element to the list
 * requires allocation of the memory for the storage in the list.

 */
template<typename T>
class CachingList {
 public:
  CachingList() {}


  bool IsEmpty() const {
    return full_data_.empty();
  }


  const T &PeekFront() const {
    if (full_data_.empty())
      throw std::out_of_range("The list is empty");

    return full_data_.front();
  }


  std::list<T> PopFront() {
    if (full_data_.empty())
      throw std::out_of_range("The list is empty");

    std::list<T> tmp;
    tmp.splice(tmp.begin(), full_data_, full_data_.begin());
    return tmp;
  }


  void Recycle(std::list<T> &elm) {
    empty_data_.splice(empty_data_.end(), elm, elm.begin(), elm.end());
  }

  void Recycle(std::list<T> &&elm) {
    Recycle(elm);
  }

  template <typename CreateDefault>
  std::list<T> GetEmpty(CreateDefault &&create_default) {
    std::list<T> tmp;
    if (empty_data_.empty()) {
      tmp.emplace_back(create_default());
    } else {
      tmp.splice(tmp.begin(), empty_data_, empty_data_.begin());
    }
    return tmp;
  }

  std::list<T> GetEmpty() {
    return GetEmpty([]() { return T(); });
  }

  void PushBack(std::list<T> &elm) {
    full_data_.splice(full_data_.end(), elm, elm.begin(), elm.end());
  }

  void PushBack(std::list<T> &&elm) {
    PushBack(elm);
  }

 private:
  std::list<T> full_data_;
  std::list<T> empty_data_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_CACHING_LIST_H_
