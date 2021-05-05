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

#ifndef DALI_CORE_MM_MEMORY_H_
#define DALI_CORE_MM_MEMORY_H_

#include <memory>
#include <utility>
#include "dali/core/mm/default_resources.h"

namespace dali {
namespace mm {

static const cudaStream_t host_sync = ((cudaStream_t)42);  // NOLINT

struct Deleter {
  void *resource;
  size_t size, alignment;
  void (*free)(void *resource, void *memory, size_t size, size_t alignment);

  void operator()(void *memory) const {
    if (memory) free(resource, memory, size, alignment);
  }
};

struct AsyncDeleter {
  void *resource;
  size_t size, alignment;
  cudaStream_t release_on_stream;
  void (*free)(void *resource, void *memory, size_t size, size_t alignment, cudaStream_t stream);

  void operator()(void *memory) const {
    if (memory) free(resource, memory, size, alignment, release_on_stream);
  }
};

template <memory_kind kind, typename Context>
Deleter GetDeleter(memory_resource<kind, Context> *resource, size_t size, size_t alignment) {
  Deleter del;
  del.resource = static_cast<void *>(resource);
  del.size = size;
  del.alignment = alignment;
  del.free = [](void *res_vptr, void *mem, size_t sz, size_t align) {
    static_cast<memory_resource<kind, Context>*>(res_vptr)->deallocate(mem, sz, align);
  };
  return del;
}

template <memory_kind kind>
AsyncDeleter GetDeleter(async_memory_resource<kind> *resource,
                   size_t size, size_t alignment, cudaStream_t stream) {
  AsyncDeleter del;
  del.resource = static_cast<void *>(resource);
  del.size = size;
  del.alignment = alignment;
  del.release_on_stream = stream;
  del.free = [](void *res_vptr, void *mem, size_t sz, size_t align, cudaStream_t s) {
    auto *rsrc = static_cast<async_memory_resource<kind>*>(res_vptr);
    if (s != host_sync) {
      rsrc->deallocate_async(mem, sz, align, s);
    } else {
      rsrc->deallocate(mem, sz, align);
    }
  };
  return del;
}

template <typename T>
using DALIUniquePtr = std::unique_ptr<T, Deleter>;

template <typename T>
using DALIAsyncUniquePtr = std::unique_ptr<T, AsyncDeleter>;

template <typename T>
void set_dealloc_stream(DALIAsyncUniquePtr<T> &ptr, cudaStream_t stream) {
  ptr.get_deleter().release_on_stream = stream;
}

/**
 * @brief Allocates uninitialized storage of size `bytes` with requested `alignment`.
 *
 * The memory is obtained from memory resource `mr`.
 * The return value is a pointer-deleter pair that can be used for building smart pointers.
 *
 * @param mr        Memory resources to allocate the memory from.
 * @param bytes     Size, in bytes, of the memory being allocated
 * @param alignment Alignment of the requested memory
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <memory_kind kind, typename Context>
std::pair<void*, Deleter> alloc_raw(memory_resource<kind, Context> *mr,
                                    size_t bytes, size_t alignment) {
  void *mem = mr->allocate(bytes, alignment);
  return { mem, GetDeleter(mr, bytes, alignment) };
}


/**
 * @brief Allocates uninitialized storage of size `bytes` with requested `alignment`.
 *
 * The memory is obtained from memory resource `mr`.
 * The return value is a pointer-deleter pair that can be used for building smart pointers.
 *
 * @param bytes     Size, in bytes, of the memory being allocated
 * @param alignment Alignment of the requested memory
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <memory_kind kind>
std::pair<void*, Deleter> alloc_raw(size_t bytes, size_t alignment) {
  auto *mr = GetDefaultResource<kind>();
  void *mem = mr->allocate(bytes, alignment);
  return { mem, GetDeleter(mr, bytes, alignment) };
}


/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr`.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr        Memory resources to allocate the memory from.
 * @param count     Number of objects for which the storage should suffice.
 * @tparam T        Type of the object for which the storage is allocated.
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <typename T, memory_kind kind, typename Context>
DALIUniquePtr<T> alloc_raw_unique(memory_resource<kind, Context> *mr, size_t count) {
  size_t bytes = sizeof(T) * count;
  size_t alignment = alignof(T);
  auto mem_del = alloc_raw(mr, bytes, alignment);
  return DALIUniquePtr<T>(static_cast<T*>(mem_del.first), std::move(mem_del.second));
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *
 * The memory is obtained from default memory resource.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param count     Number of objects for which the storage should suffice.
 * @tparam T        Type of the object for which the storage is allocated.
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <typename T, memory_kind kind>
auto alloc_raw_unique(size_t count) {
  return alloc_raw_unique<T>(GetDefaultResource<kind>(), count);
}


/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr`.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a shared pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr        Memory resources to allocate the memory from.
 * @param count     Number of objects for which the storage should suffice.
 * @tparam T        Type of the object for which the storage is allocated.
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <typename T, memory_kind kind, typename Context>
std::shared_ptr<T> alloc_raw_shared(memory_resource<kind, Context> *mr, size_t count) {
  size_t bytes = sizeof(T) * count;
  size_t alignment = alignof(T);
  auto mem_del = alloc_raw(mr, bytes, alignment);
  return std::shared_ptr<T>(static_cast<T*>(mem_del.first), std::move(mem_del.second));
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *
 * The memory is obtained from default memory resource.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a shared pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param count     Number of objects for which the storage should suffice.
 * @tparam T        Type of the object for which the storage is allocated.
 * @tparam kind     The kind of requested memory.
 * @tparam Context  The execution context in which the memory will be available.
 */
template <typename T, memory_kind kind>
auto alloc_raw_shared(size_t count) {
  return alloc_raw_shared<T>(GetDefaultResource<kind>(), count);
}

/**
 * @brief Allocates uninitialized storage of size `bytes` with requested `alignment`
 *        with stream-ordered semantics
 *
 * This function allocates memory with stream-ordered semantics. The `alloc_stream` denotes
 * the stream on which the memory can be safely used without additional synchronization.
 * Use the value `host_sync` if the memory needs to be accessible on all streams or on host as
 * soon as the function returns.
 * `dealloc_stream` denotest the stream for deallocation. Stream-ordered semantics guarantee,
 * that if there's still some work pending on `dealloc_stream`, it will finish before the memory
 * returned by this function is freed. Use `host_sync` for host-synchronous execution.
 *
 * The memory is obtained from memory resource `mr` with stream semantics.
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr              Memory resources to allocate the memory from
 * @param count           Number of objects for which the storage should suffice
 * @param alloc_stream    The CUDA stream on which the memory is immediately usable
 * @param dealloc_stream  The CUDA stream which is guaranteed to finish all work scheduled
 *                        before the deallocation of the memory.
 * @tparam kind           The kind of requested memory.
 */
template <memory_kind kind>
std::pair<void*, AsyncDeleter> alloc_raw_async(async_memory_resource<kind> *mr,
                                    size_t bytes,
                                    size_t alignment,
                                    cudaStream_t alloc_stream,
                                    cudaStream_t dealloc_stream) {
  void *mem = alloc_stream == host_sync
    ? mr->allocate(bytes, alignment)
    : mr->allocate_async(bytes, alignment, alloc_stream);
  return { mem, GetDeleter(mr, bytes, alignment, dealloc_stream) };
}

template <memory_kind kind>
auto alloc_raw_async(size_t bytes,
                     size_t alignment,
                     cudaStream_t alloc_stream,
                     cudaStream_t dealloc_stream) {
  return alloc_raw_async<kind>(GetDefaultResource<kind>(), bytes, alignment,
                               alloc_stream, dealloc_stream);
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *        with stream-ordered semantics
 *
 * This function allocates memory with stream-ordered semantics. The `alloc_stream` denotes
 * the stream on which the memory can be safely used without additional synchronization.
 * Use the value `host_sync` if the memory needs to be accessible on all streams or on host as
 * soon as the function returns.
 * `dealloc_stream` denotest the stream for deallocation. Stream-ordered semantics guarantee,
 * that if there's still some work pending on `dealloc_stream`, it will finish before the memory
 * returned by this function is freed. Use `host_sync` for host-synchronous execution.
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr` with stream semantics.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr              Memory resources to allocate the memory from
 * @param count           Number of objects for which the storage should suffice
 * @param alloc_stream    The CUDA stream on which the memory is immediately usable
 * @param dealloc_stream  The CUDA stream which is guaranteed to finish all work scheduled
 *                        before the deallocation of the memory.
 * @tparam T              Type of the object for which the storage is allocated.
 * @tparam kind           The kind of requested memory.
 */
template <typename T, memory_kind kind>
DALIAsyncUniquePtr<T> alloc_raw_async_unique(async_memory_resource<kind> *resource,
                                             size_t count,
                                             cudaStream_t alloc_stream,
                                             cudaStream_t dealloc_stream) {
  size_t bytes = sizeof(T) * count;
  size_t alignment = alignof(T);
  auto mem_del = alloc_raw_async(resource, bytes, alignment, alloc_stream, dealloc_stream);
  return DALIAsyncUniquePtr<T>(static_cast<T*>(mem_del.first), std::move(mem_del.second));
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *        with stream-ordered semantics
 *
 * This function allocates memory with stream-ordered semantics. The `alloc_stream` denotes
 * the stream on which the memory can be safely used without additional synchronization.
 * Use the value `host_sync` if the memory needs to be accessible on all streams or on host as
 * soon as the function returns.
 * `dealloc_stream` denotest the stream for deallocation. Stream-ordered semantics guarantee,
 * that if there's still some work pending on `dealloc_stream`, it will finish before the memory
 * returned by this function is freed. Use `host_sync` for host-synchronous execution.
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr` with stream semantics.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a unique pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param count           Number of objects for which the storage should suffice
 * @param alloc_stream    The CUDA stream on which the memory is immediately usable
 * @param dealloc_stream  The CUDA stream which is guaranteed to finish all work scheduled
 *                        before the deallocation of the memory.
 * @tparam T              Type of the object for which the storage is allocated.
 * @tparam kind           The kind of requested memory.
 */
template <typename T, memory_kind kind>
auto alloc_raw_async_unique(size_t count,
                            cudaStream_t alloc_stream,
                            cudaStream_t dealloc_stream) {
  return alloc_raw_async_unique<T>(GetDefaultResource<kind>(),
                                   count, alloc_stream, dealloc_stream);
}



/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *        with stream-ordered semantics
 *
 * This function allocates memory with stream-ordered semantics. The `alloc_stream` denotes
 * the stream on which the memory can be safely used without additional synchronization.
 * Use the value `host_sync` if the memory needs to be accessible on all streams or on host as
 * soon as the function returns.
 * `dealloc_stream` denotest the stream for deallocation. Stream-ordered semantics guarantee,
 * that if there's still some work pending on `dealloc_stream`, it will finish before the memory
 * returned by this function is freed. Use `host_sync` for host-synchronous execution.
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr` with stream semantics.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a shared pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param mr              Memory resources to allocate the memory from
 * @param count           Number of objects for which the storage should suffice
 * @param alloc_stream    The CUDA stream on which the memory is immediately usable
 * @param dealloc_stream  The CUDA stream which is guaranteed to finish all work scheduled
 *                        before the deallocation of the memory.
 * @tparam T              Type of the object for which the storage is allocated.
 * @tparam kind           The kind of requested memory.
 */
template <typename T, memory_kind kind>
std::shared_ptr<T> alloc_raw_async_shared(async_memory_resource<kind> *resource,
                                             size_t count,
                                             cudaStream_t alloc_stream,
                                             cudaStream_t dealloc_stream) {
  size_t bytes = sizeof(T) * count;
  size_t alignment = alignof(T);
  auto mem_del = alloc_raw_async(resource, bytes, alignment, alloc_stream, dealloc_stream);
  return std::shared_ptr<T>(static_cast<T*>(mem_del.first), std::move(mem_del.second));
}

/**
 * @brief Allocates uninitialized storage for `count` objects of type `T`
 *        with stream-ordered semantics
 *
 * This function allocates memory with stream-ordered semantics. The `alloc_stream` denotes
 * the stream on which the memory can be safely used without additional synchronization.
 * Use the value `host_sync` if the memory needs to be accessible on all streams or on host as
 * soon as the function returns.
 * `dealloc_stream` denotest the stream for deallocation. Stream-ordered semantics guarantee,
 * that if there's still some work pending on `dealloc_stream`, it will finish before the memory
 * returned by this function is freed. Use `host_sync` for host-synchronous execution.
 *
 * This function allocates raw memory with suitable size and alignment to accommodate `count`
 * object of type `T`. The memory is obtained from memory resource `mr` with stream semantics.
 * The size of memory is at least sizeof(T) * count and the alignment is at least alignof(T).
 * The return value is a shared pointer with a deleter that will safely dispose of the allocated
 * buffer, but does not call destructors of the objects.
 *
 * @param count           Number of objects for which the storage should suffice
 * @param alloc_stream    The CUDA stream on which the memory is immediately usable
 * @param dealloc_stream  The CUDA stream which is guaranteed to finish all work scheduled
 *                        before the deallocation of the memory.
 * @tparam T              Type of the object for which the storage is allocated.
 * @tparam kind           The kind of requested memory.
 */
template <typename T, memory_kind kind>
auto alloc_raw_async_shared(size_t count,
                            cudaStream_t alloc_stream,
                            cudaStream_t dealloc_stream) {
  return alloc_raw_async_shared<T>(GetDefaultResource<kind>(),
                                   count, alloc_stream, dealloc_stream);
}



}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_MEMORY_H_
