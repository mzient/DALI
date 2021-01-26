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

#ifndef DALI_KERNELS_COMMON_DISJOINT_SET_H_
#define DALI_KERNELS_COMMON_DISJOINT_SET_H_

#include <type_traits>
#include "dali/core/span.h"

namespace dali {
namespace kernels {

template <typename T, typename GroupId>
struct group_ops {
  static_assert(std::is_convertible<T, GroupId>::value && std::is_convertible<GroupId, T>::value,
                "This implementation of group ops requires that the element and index are "
                "convertible to each other.");

  static inline GroupId get_group(const T &x) {
    return x;
  }

  static inline GroupId set_group(T &x, GroupId new_id) {
    GroupId old = x;
    x = new_id;
    return old;
  }
};

template <typename T, typename GroupId = int, typename Ops = group_ops<T, GroupId>>
struct disjoint_set {
  template <typename Container>
  void init(Container &&items, GroupId start_index = {}) {
    for (auto &x : items) {
      Ops::set_group(x, start_index);
      ++start_index;
    }
  }

  void init(T *items, int n, GroupId start_index = {}) {
    init(make_span(items, n, start_index));
  }

  template <class Container>
  static inline GroupId find(Container &&items, GroupId x) {
    GroupId x0 = x;

    // find the label
    for (;;) {
      GroupId g = Ops::get_group(items[x]);
      if (g == x)
        break;
      x = g;
    }

    GroupId r = x;

    // assign all intermediate labels to save time in subsequent calls
    x = x0;
    while (x != Ops::get_group(items[x])) {
      x0 = Ops::set_group(items[x], r);
      x = x0;
    }

    return r;
  }

  template <typename Container>
  static inline GroupId merge(Container &&items, GroupId x, GroupId y) {
    y = find(std::forward<Container>(items), y);
    x = find(std::forward<Container>(items), x);
    if (x < y) {
      Ops::set_group(items[y], x);
      return x;
    } else if (y < x) {
      Ops::set_group(items[x], y);
      return y;
    } else {
      // already merged
      return x;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_DISJOINT_SET_H_

