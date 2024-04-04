// Copyright (c) 2022, 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/caching_list.h"
#include <gtest/gtest.h>
#include <utility>

namespace dali::test {

namespace {

template<typename T>
struct TestType {
  using element_type = T;
  T val;

  bool operator==(const T &other) const {
    return other == val;
  }
};

}  // namespace


TEST(CachingListTest, ListTest) {
  CachingList<TestType<int>> cl;

  auto push = [&](int val) {
      auto elem = cl.GetEmpty([]() { return TestType<int>(); });
      elem.front() = { val };
      cl.PushBack(std::move(elem));
  };

  EXPECT_THROW(cl.PeekFront(), std::out_of_range);
  push(6);
  EXPECT_EQ(cl.PeekFront(), 6);
  push(9);
  EXPECT_EQ(cl.PeekFront(), 6);
  cl.PopFront();
  EXPECT_EQ(cl.PeekFront(), 9);
  push(13);
  EXPECT_EQ(cl.PeekFront(), 9);
  cl.Recycle(cl.PopFront());
  EXPECT_EQ(cl.PeekFront(), 13);
  push(42);
  EXPECT_EQ(cl.PeekFront(), 13);
  push(69);
  EXPECT_EQ(cl.PeekFront(), 13);
  cl.Recycle(cl.PopFront());
  EXPECT_EQ(cl.PeekFront(), 42);
  cl.Recycle(cl.PopFront());
  EXPECT_EQ(cl.PeekFront(), 69);
  cl.Recycle(cl.PopFront());
  EXPECT_THROW(cl.PeekFront(), std::out_of_range);
  push(666);
  EXPECT_EQ(cl.PeekFront(), 666);
  push(1337);
  EXPECT_EQ(cl.PeekFront(), 666);
  cl.Recycle(cl.PopFront());
  EXPECT_EQ(cl.PeekFront(), 1337);
  cl.Recycle(cl.PopFront());
  EXPECT_THROW(cl.PeekFront(), std::out_of_range);
  push(1234);
  EXPECT_EQ(cl.PeekFront(), 1234);
  push(4321);
  EXPECT_EQ(cl.PeekFront(), 1234);
  cl.Recycle(cl.PopFront());
  EXPECT_EQ(cl.PeekFront(), 4321);
  cl.Recycle(cl.PopFront());
  EXPECT_THROW(cl.PeekFront(), std::out_of_range);
  EXPECT_THROW(cl.PopFront(), std::out_of_range);
}
}  // namespace dali::test
