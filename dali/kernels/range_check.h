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

#include <stdlib.h>
#include <stdexcept>
#include <cstdint>

namespace dali {
namespace detail {
inline void raise_out_of_range(long long value, long long lo, long long hi) {
  char buf[128];
  snprintf(buf, sizeof(buf), "Value %lld out of range [%lld..%lld)", value, lo, hi);
  throw std::out_of_range(buf);
}
}  // namespace detail

inline void range_check(long long value, long long lo, long long hi) {
  if (value < lo || value >= hi)
    detail::raise_out_of_range(value, lo, hi);
}

}  // namespace dali
