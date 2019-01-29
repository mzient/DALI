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

#include <cuda_runtime.h>
#include <cassert>
#include "dali/kernels/alloc.h"
#include "dali/kernels/static_switch.h"
#include "dali/kernels/gpu_utils.h"

namespace dali {
namespace kernels {
namespace memory {

template <AllocType>
struct Allocator;

template <>
struct Allocator<AllocType::Host> {
  static std::function<void(void*)> GetDeallocator() {
    return free;
  }
  static void *Allocate(size_t bytes) { return malloc(bytes); }
};

template <>
struct Allocator<AllocType::Pinned> {
  static std::function<void(void*)> GetDeallocator() {
    int dev = 0;
    cudaGetDevice(&dev);
    return [dev](void *ptr) {
      DeviceGuard guard(dev);
      cudaFreeHost(ptr);
    };
  }

  static void *Allocate(size_t bytes) {
    void *ptr = nullptr;
    cudaMallocHost(&ptr, bytes);
    return ptr;
  }
};

template <>
struct Allocator<AllocType::GPU> {
  static std::function<void(void*)> GetDeallocator() {
    int dev = 0;
    cudaGetDevice(&dev);
    return [dev](void *ptr) {
      DeviceGuard guard(dev);
      cudaFree(ptr);
    };
  }

  static void *Allocate(size_t bytes) {
    void *ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
  }
};


template <>
struct Allocator<AllocType::Unified> {
  static std::function<void(void*)> GetDeallocator() {
    int dev = 0;
    cudaGetDevice(&dev);
    return [dev](void *ptr) {
      DeviceGuard guard(dev);
      cudaFree(ptr);
    };
  }

  static void *Allocate(size_t bytes) {
    void *ptr = nullptr;
    cudaMallocManaged(&ptr, bytes);
    return ptr;
  }
};


std::function<void(void*)> GetDeallocator(AllocType type) {
  VALUE_SWITCH(type, type_label,
    (AllocType::Host, AllocType::Pinned, AllocType::GPU, AllocType::Unified),
    (return Allocator<type_label>::GetDeallocator()),
    (assert(!"Invalid allocation type requested");)
  );  // NOLINT
}

void *Allocate(AllocType type, size_t size) {
  VALUE_SWITCH(type, type_label,
    (AllocType::Host, AllocType::Pinned, AllocType::GPU, AllocType::Unified),
    (return Allocator<type_label>::Allocate(size)),
    (assert(!"Invalid allocation type requested");
    return nullptr;)
  );  // NOLINT
}

}  // namespace memory
}  // namespace kernels
}  // namespace dali
