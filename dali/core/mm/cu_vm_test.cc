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

#include "dali/core/mm/cu_vm.h"
#include <gtest/gtest.h>

#if DALI_USE_CUDA_VM_MAP

namespace dali {
namespace mm {
namespace test {

TEST(CUMemAddressRange, Reserve) {
  ASSERT_TRUE(cuInitChecked());
  int64_t requested = 4000000;
  cuvm::CUMemAddressRange range = cuvm::CUMemAddressRange::Reserve(requested);
  EXPECT_GE(static_cast<int64_t>(range.size()), requested);
  CUpointer_attribute attrs[3] = {
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    CU_POINTER_ATTRIBUTE_MAPPED
  };
  CUdeviceptr start = 0;
  size_t size = 0;
  bool mapped = false;
  void *data[3] = { &start, &size, &mapped };
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, range.ptr() + 100));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_FALSE(mapped);
}

TEST(CUMemAddressRange, ReserveAndMap) {
  ASSERT_TRUE(cuInitChecked());
  int64_t virt_size = 10<<20;
  int64_t phys_size = 4<<20;
  cuvm::CUMemAddressRange range = cuvm::CUMemAddressRange::Reserve(virt_size);
  cuvm::CUMem mem = cuvm::CUMem::Create(phys_size);
  CUdeviceptr base = range.ptr();
  EXPECT_GE(static_cast<int64_t>(range.size()), virt_size);
  CUpointer_attribute attrs[3] = {
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    CU_POINTER_ATTRIBUTE_MAPPED
  };
  CUdeviceptr start = 0;
  size_t size = 0;
  bool mapped = false;
  void *data[3] = { &start, &size, &mapped };
  void *ptr = cuvm::Map(base, mem);
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, base + 100));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_TRUE(mapped);
  cuvm::Unmap(ptr, mem.size());
  CUDA_CALL(cuPointerGetAttributes(3, attrs, data, base + 1234));
  EXPECT_EQ(start, range.ptr());
  EXPECT_EQ(size, range.size());
  EXPECT_FALSE(mapped);
}

}  // namespace test
}  // namespace mm
}  // namespace dali


#endif
