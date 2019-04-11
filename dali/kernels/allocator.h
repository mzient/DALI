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

#ifndef DALI_KERNELS_ALLOCATOR_H_
#define DALI_KERNELS_ALLOCATOR_H_

#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>

namespace dali {

struct BasicAllocator {
  virtual void *allocate(size_t size, size_t alicnment) = 0;
  virtual void deallocate(void *ptr, size_t size) = 0;
};

inline void *align(std::size_t alignment, std::size_t size, void *&ptr, std::size_t &space) {
  std::uintptr_t unaligned = reinterpret_cast<std::uintptr_t>(ptr);
  std::size_t padding = (-unaligned) & (alignment-1);

  if (space < size + padding)
    return nullptr;

  space -= padding;
  return ptr = reinterpret_cast<void *>(unaligned + padding);
}

struct MemBlock {
  void *memory;
  size_t usage;
  size_t capacity;

  bool owns(void *ptr) const {
    std::ptrdiff_t offset = static_cast<char *>(ptr) - static_cast<char*>(memory);
    return offset >= 0 && offset < capacity;
  }

  void *allocate(size_t size, size_t alignment) {
    size_t space = capacity - usage;
    void *ptr = static_cast<char*>(memory) + usage;
    if (align(alignment, size, ptr, space)) {
      space -= size;
      usage = capacity - space;
      return ptr;
    }
    return nullptr;
  }

  void deallocate(void *ptr, size_t size) {
    std::ptrdiff_t offset = static_cast<char *>(ptr) - static_cast<char *>(memory);
    assert(offset >= 0 && offset < usage && "Block not owned by this allocator");
    if (offset + size == static_cast<std::ptrdiff_t>(usage)) {
      usage -= size;  // we lose alignment space, if there was any :(
    }  // otherwise we can't free, but who cares
  }

  void free_all() {
    usage = 0;
  }
};

class MemPool {
 public:
  BasicAllocator *alloc = nullptr;

  MemPool(BasicAllocator *alloc) : alloc(alloc) {
    blocks.reserve(64);
  }

  ~MemPool() {
    for (auto &b : blocks) {
      free_block(b);
    }
  }

  void set_estimate(size_t size) {
    if (size > max_estimate)
      max_estimate = size;
  }

  void *allocate(size_t size, size_t alignment) {
    if (alignment > max_alignment)
      max_alignment = alignment;

    for (auto &b : blocks) {
      void *mem = b.allocate(size, alignment);
      if (mem) {
        total_size += size;
        if (total_size > max_size)
          max_size = total_size;
        return mem;
      }
    }
    size_t new_block_size = std::max(max_estimate, size);
    if (!blocks.empty())
      new_block_size = std::max(new_block_size, next_size(blocks.back().capacity));

    auto blk = alloc_block(new_block_size, alignment);
    if (!blk.memory)
      return nullptr;
    blocks.push_back(blk);
    void *ret = blocks.back().allocate(size, alignment);

    if (!ret)
      return nullptr;

    total_size += size;
    if (total_size > max_size)
      max_size = total_size;

    return ret;
  }

  void deallocate(void *ptr, size_t size) {
    for (auto &b : blocks) {
      if (b.owns(ptr)) {
        b.deallocate(ptr, size);
        return;
      }
    }
    assert(!"Pointer not owned by the allocator");
  }

  bool owns(void *ptr) const {
    for (auto &b : blocks) {
      if (b.owns(ptr))
        return true;
    }
    return false;
  }

  void free_all() {
    for (auto &block : blocks) {
      block.free_all();
    }
    //collapse();
    total_size = 0;
  }

private:

  static size_t next_size(size_t s) {
    if (s < 16<<20) {           // 16 MB
      return s * 2;
    } else if (s < 256<<20) {   // 256 MB
      return s * 1.5;
    } else {
      return s * 1.25;
    }
  }

  void collapse() {
    int n = blocks.size();
    if (n < 1)
      return;
    for (int i = 0; i < n-1; i++) {
      free_block(blocks[i]);
    }
    if (blocks[n-1].capacity >= max_size) {
      std::swap(blocks[0], blocks[n-1]);
      blocks.resize(1);
      return;
    }
    free_block(blocks[n-1]);
    blocks.resize(1);
    blocks[0] = alloc_block(max_size, max_alignment);
  }

  MemBlock alloc_block(size_t size, size_t alignment) {
    MemBlock blk;
    alignment = std::max(alignment, block_alignment);
    blk.memory = alloc->allocate(size, alignment);
    blk.capacity = size;
    blk.usage = 0;
    return blk;
  }

  void free_block(MemBlock &block) {
    alloc->deallocate(block.memory, block.capacity);
    block.memory = nullptr;
  }

  size_t min_block_size = 1<<16;
  size_t max_seen_alloc = 0;

  std::vector<MemBlock> blocks;

  size_t max_size = 0;
  size_t total_size = 0;
  size_t capacity = 0;
  size_t max_estimate = 0;
  size_t max_alignment = block_alignment;
  static constexpr size_t block_alignment = 256;
};

constexpr size_t MemPool::block_alignment;

template <typename T>
struct scratchpad_allocator {
  using value_type = T;
  using reference = T&;
  using pointer = T*;
  using const_pointer = const T*;
  using void_pointer = void*;

  scratchpad_allocator() {}

  scratchpad_allocator(MemPool &mem) : mem(&mem) {}
  template <typename U>
  scratchpad_allocator(const scratchpad_allocator &other) : mem(other.mem) {}

  MemPool *mem = nullptr;
  T *allocate(size_t n) {
    return reinterpret_cast<T*>(mem->allocate(n * sizeof(T), alignof(T)));
  }

  void deallocate(T *ptr, size_t n) noexcept {
    mem->deallocate(ptr, n * sizeof(T));
  }
};

struct Mallocator : BasicAllocator {
  void *allocate(size_t bytes, size_t alignment) {
    bytes = -((-bytes)&(-alignment));  // it's a kind of magic..
                                       //                  magic
    return aligned_alloc(alignment, bytes);
  }

  void deallocate(void *ptr, size_t bytes) {
    free(ptr);
    (void)bytes;
  }
};

}  // namespace dali

#endif  // DALI_KERNELS_ALLOCATOR_H_
