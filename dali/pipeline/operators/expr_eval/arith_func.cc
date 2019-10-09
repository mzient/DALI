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

#include "function.h"

namespace dali {
namespace expr {

template <typename LHS, typename RHS, typename PerElementFunc>
class BinaryOpCPU_TT : FunctionImpl<CPUBackend> {
 public:
  using OutputType = promoted_t<LHS, RHS, PerElementFunc>;
  void Run(span<const Tile> arg_tiles) {
    assert(tiles.size() == 3);
    auto n = arg_tiles[0].num_elements;
    PerElementFunc f;
    auto *out = static_cast<OutputType*>(arg_tiles[0]);
    auto *lhs = static_cast<const LHS*>(arg_tiles[1]);
    auto *rhs = static_cast<const RHS*>(arg_tiles[2]);
    for (int i = 0; i < tiles[0].element_size; i++) {
      out[i] = f(lhs[i], rhs[i]);
    }
  }

};

}  // namespace expr
}  // namespace dali
