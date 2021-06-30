// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include "dali/core/random.h"
#include "dali/core/mm/cuda_vm_resource.h"

namespace dali {
namespace mm {
namespace test {

class VMResourceTest : public ::testing::Test {
 public:
  void TestAlloc() {
    cuda_vm_resource res;
    res.block_size_ = 32<<20;  // fix the block size at 32 MiB for this test
    void *ptr = res.allocate(100<<20);
    void *ptr2 = res.allocate(100<<20);
    EXPECT_EQ(res.va_regions_.size(), 1u);
    EXPECT_EQ(res.va_regions_.front().mapped.find(false), 7);
    res.deallocate(ptr, 100<<20);
    EXPECT_EQ(res.va_regions_.front().available.find(true), 0);
    EXPECT_EQ(res.va_regions_.front().available.find(false), 3);
    void *ptr3 = res.allocate(100<<20);
    EXPECT_EQ(ptr, ptr3);
    cuda_vm_resource::mem_handle_t blocks[3];
    for (int i = 0; i < 3; i++)
      blocks[i] = res.va_regions_.front().mapping[i];
    res.deallocate(ptr3, 100<<20);
    // let's request more than was deallocated - it should go at the end of VA range
    void *ptr4 = res.allocate(150<<20);
    EXPECT_EQ(ptr4, static_cast<void*>(static_cast<char*>(ptr2) + (100<<20)));
    for (int i = 0; i < 3; i++) {
      // ptr4 should start 200 MiB from start, which is block 6
      // block 7 should be still unmapped, hence i+7.
      // Let's check that the 1st 3 blocks have been reused rather than newly allocated.
      EXPECT_EQ(res.va_regions_.back().mapping[i+7], blocks[i]);
    }
    res.deallocate(ptr2, 100<<20);
    res.deallocate(ptr4, 150<<20);
  }

  void MapRandomBlocks(cuda_vm_resource::va_region &region, int blocks_to_map) {
    assert(blocks_to_map < region.num_blocks());
    std::vector<int> block_idxs(region.num_blocks());
    random_permutation(block_idxs, rng_);
    block_idxs.resize(blocks_to_map);
    for (int blk_idx : block_idxs) {
      region.map_block(blk_idx, cuvm::CUMem::Create(region.block_size));
    }
  }

  struct va_region_copy : cuda_vm_resource::va_region {
    using va_region::va_region;
    va_region_copy(va_region_copy &&other) = default;
    ~va_region_copy() {
      mapping.clear();
    }
  };

  static va_region_copy Copy(const cuda_vm_resource::va_region &in) {
    va_region_copy out(in.address_range, in.block_size);
    out.available_blocks = in.available_blocks;
    out.mapping   = in.mapping;
    out.mapped    = in.mapped;
    out.available = in.available;
    return out;
  }

  void ComparePart(const cuda_vm_resource::va_region &region,
                   int pos,
                   const cuda_vm_resource::va_region &ref) {
    for (int i = 0, j = pos; i < ref.num_blocks(); i++, j++) {
      EXPECT_EQ(region.mapping[j],   ref.mapping[i])   << "@ " << j;
      EXPECT_EQ(region.mapped[j],    ref.mapped[i])    << "@ " << j;
      EXPECT_EQ(region.available[j], ref.available[i]) << "@ " << j;
    }
  }

  void TestRegionExtendAfter() {
    cuda_vm_resource res;
    res.block_size_ = 4 << 20;  // 4 MiB
    const int b1 = 32, b2 = 64;
    const size_t s1 = b1 * res.block_size_;
    const size_t s2 = b2 * res.block_size_;
    res.va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(s1 + s2));
    cuvm::CUAddressRange total = res.va_ranges_.back();
    cuvm::CUAddressRange part1 = { total.ptr(),           s1 };
    cuvm::CUAddressRange part2 = { total.ptr() + s1,      s2 };
    res.va_add_region(part1);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    MapRandomBlocks(res.va_regions_[0], 10);
    va_region_copy va1 = Copy(res.va_regions_[0]);
    res.va_add_region(part2);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    auto &region = res.va_regions_.back();
    ASSERT_EQ(region.num_blocks(), b1 + b2);
    ComparePart(region, 0, va1);
    for (int i = b1; i < b1 + b2; i++) {
      EXPECT_FALSE(region.mapped[i]);
      EXPECT_FALSE(region.mapped[i]);
    }
  }

  void TestRegionExtendBefore() {
    cuda_vm_resource res;
    res.block_size_ = 4 << 20;  // 4 MiB
    const int b1 = 32, b2 = 64;
    const size_t s1 = b1 * res.block_size_;
    const size_t s2 = b2 * res.block_size_;
    res.va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(s1 + s2));
    cuvm::CUAddressRange total = res.va_ranges_.back();
    cuvm::CUAddressRange part1 = { total.ptr(),           s1 };
    cuvm::CUAddressRange part2 = { total.ptr() + s1,      s2 };
    res.va_add_region(part2);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    MapRandomBlocks(res.va_regions_[0], 10);
    va_region_copy va1 = Copy(res.va_regions_[0]);
    res.va_add_region(part1);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    auto &region = res.va_regions_.back();
    ASSERT_EQ(region.num_blocks(), b1 + b2);
    ComparePart(region, b1, va1);
    for (int i = 0; i < b1; i++) {
      EXPECT_FALSE(region.mapped[i]);
      EXPECT_FALSE(region.mapped[i]);
    }
  }

  void TestRegionMerge() {
    cuda_vm_resource res;
    res.block_size_ = 4 << 20;  // 4 MiB
    const int b1 = 32, b2 = 64, b3 = 32;
    const size_t s1 = b1 * res.block_size_;
    const size_t s2 = b2 * res.block_size_;
    const size_t s3 = b3 * res.block_size_;
    res.va_ranges_.push_back(cuvm::CUMemAddressRange::Reserve(s1 + s2 + s3));
    cuvm::CUAddressRange total = res.va_ranges_.back();
    cuvm::CUAddressRange part1 = { total.ptr(),           s1 };
    cuvm::CUAddressRange part2 = { total.ptr() + s1,      s2 };
    cuvm::CUAddressRange part3 = { total.ptr() + s1 + s2, s3 };
    res.va_add_region(part1);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    MapRandomBlocks(res.va_regions_[0], 10);
    va_region_copy va1 = Copy(res.va_regions_[0]);
    res.va_add_region(part3);
    ASSERT_EQ(res.va_regions_.size(), 2u);
    MapRandomBlocks(res.va_regions_[1], 12);
    va_region_copy va3 = Copy(res.va_regions_[1]);
    res.va_add_region(part2);
    ASSERT_EQ(res.va_regions_.size(), 1u);
    auto &region = res.va_regions_.back();
    ASSERT_EQ(region.num_blocks(), b1 + b2 + b3);
    ComparePart(region, 0, va1);
    ComparePart(region, b1 + b2, va3);
    for (int i = b1; i < b1 + b2; i++) {
      EXPECT_FALSE(region.mapped[i]);
      EXPECT_FALSE(region.mapped[i]);
    }
  }

  std::mt19937_64 rng_{12345};
};

TEST_F(VMResourceTest, BasicTest) {
  this->TestAlloc();
}

TEST_F(VMResourceTest, RegionMerge) {
  this->TestRegionMerge();
}

TEST_F(VMResourceTest, RegionExtendAfter) {
  this->TestRegionExtendAfter();
}

TEST_F(VMResourceTest, RegionExtendBefore) {
  this->TestRegionExtendBefore();
}

}  // namespace test
}  // namespace mm
}  // namespace dali


