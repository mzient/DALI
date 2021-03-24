// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <random>
#include "dali/core/mm/async_pool.h"
#include "dali/core/mm/mm_test_utils.h"
#include "dali/core/cuda_stream.h"
#include "rmm/mr/device/pool_memory_resource.hpp"

namespace dali {
namespace mm {

struct GPUHog {
  GPUHog() {
    CUDA_CALL(cudaMalloc(&mem, size));
  }
  ~GPUHog() {
    CUDA_DTOR_CALL(cudaFree(mem));
  }

  void run(cudaStream_t stream, int count = 1) {
    for (int i = 0; i < count; i++) {
      CUDA_CALL(cudaMemsetAsync(mem, i+1, size, stream));
    }
  }

  uint8_t *mem;
  size_t size = 16<<20;
};

TEST(MMAsyncPool, SingleStreamReuse) {
  GPUHog hog;

  CUDAStream stream = CUDAStream::Create(true);
  test::test_device_resource upstream;

  async_pool_base<memory_kind::device, free_tree, std::mutex> pool(&upstream);
  stream_view sv(stream);
  int size1 = 1<<20;
  void *ptr = pool.allocate_async(size1, sv);
  hog.run(stream, 2);
  pool.deallocate_async(ptr, size1, sv);
  void *p2 = pool.allocate_async(size1, sv);
  CUDA_CALL(cudaStreamSynchronize(stream));
  EXPECT_EQ(ptr, p2);
}


namespace {

struct block {
  void *ptr;
  size_t size;
};

template <typename Pool, typename Mutex>
void AsyncPoolTest(Pool &pool, vector<block> &blocks, Mutex &mtx, CUDAStream &stream) {
  stream_view sv(stream);
  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<> size_dist(100, 10000);
  std::bernoulli_distribution action_dist;
  for (int i = 0; i < 100000; i++) {
    if (action_dist(rng) || blocks.empty()) {
      size_t size = size_dist(rng);
      void *ptr = pool.allocate_async(size, sv);
      {
        std::lock_guard<Mutex> guard(mtx);
        blocks.push_back({ ptr, size });
      }
    } else {
      block blk;
      {
        std::lock_guard<Mutex> guard(mtx);
        if (blocks.empty())
          continue;
        int i = std::uniform_int_distribution<>(0, blocks.size()-1)(rng);
        std::swap(blocks[i], blocks.back());
        blk = blocks.back();
        blocks.pop_back();
      }
      pool.deallocate_async(blk.ptr, blk.size, sv);
    }
  }
}

}  // namespace std

TEST(MMAsyncPool, SingleStreamRandom) {
  CUDAStream stream = CUDAStream::Create(true);
  test::test_device_resource upstream;

  vector<block> blocks;

  async_pool_base<memory_kind::device, free_tree, std::mutex> pool(&upstream);
  detail::dummy_lock mtx;
  AsyncPoolTest(pool, blocks, mtx, stream);


  CUDA_CALL(cudaStreamSynchronize(stream));
}

TEST(MMAsyncPool, MultiThreadedSingleStreamRandom) {
  CUDAStream stream = CUDAStream::Create(true);
  mm::test::test_device_resource upstream;
  {
    vector<block> blocks;
    std::mutex mtx;

    async_pool_base<memory_kind::device, free_tree, std::mutex> pool(&upstream);

    vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&, t]() {
        AsyncPoolTest(pool, blocks, mtx, stream);
      }));
    }
    for (auto &t : threads)
      t.join();

  }
  CUDA_CALL(cudaStreamSynchronize(stream));
  upstream.check_leaks();
}

TEST(MMAsyncPool, MultiThreadedMultiStreamRandom) {
  mm::test::test_device_resource upstream;
  {
    vector<block> blocks;
    std::mutex mtx;

    async_pool_base<memory_kind::device, free_tree, std::mutex> pool(&upstream);

    vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
      threads.push_back(std::thread([&, t]() {
        CUDAStream stream = CUDAStream::Create(true);
        AsyncPoolTest(pool, blocks, mtx, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
      }));
    }
    for (auto &t : threads)
      t.join();

  }
  upstream.check_leaks();
}


}  // namespace mm
}  //  dali
