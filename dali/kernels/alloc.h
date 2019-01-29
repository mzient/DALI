// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_ALLOC_H_
#define DALI_KERNELS_ALLOC_H_

#include <functional>
#include <memory>
#include <type_traits>
#include "dali/kernels/alloc_type.h"

namespace dali {
namespace kernels {
namespace memory {

std::function<void(void*)> GetDeallocator(AllocType type);
void *Allocate(AllocType type, size_t size);

template <typename T>
std::shared_ptr<T> alloc_shared(AllocType type, size_t count) {
  static_assert(std::is_pod<T>::value, "Only POD types are supported");
  return { reinterpret_cast<T*>(Allocate(type, count*sizeof(T))), GetDeallocator(type) };
}

template <typename T>
std::unique_ptr<T, std::function<void(void*)>> alloc_unique(AllocType type, size_t count) {
  static_assert(std::is_pod<T>::value, "Only POD types are supported");
  return { reinterpret_cast<T*>(Allocate(type, count*sizeof(T))), GetDeallocator(type) };
}

}  // namespace memory
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_ALLOC_H_
