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

#ifndef DALI_KERNELS_SLICE_SLICE_ARGS_H_
#define DALI_KERNELS_SLICE_SLICE_ARGS_H_

#include <stdint.h>
#include <array>

namespace dali {
namespace kernels {

template <size_t Dims>
struct SliceArgs {
  std::array<int64_t, Dims> anchor;
  std::array<int64_t, Dims> shape;
};

template <size_t Dims>
struct SliceFlipNormalizePermuteArgs {
  template <typename Shape>
  explicit SliceFlipNormalizePermuteArgs(const Shape &_shape) {
    for (size_t d = 0; d < Dims; d++) {
      anchor[d] = 0;
      shape[d] = _shape[d];
      padded_shape[d] = _shape[d];
      flip[d] = false;
      permuted_dims[d] = d;
    }
  }

  std::array<int64_t, Dims> anchor;
  std::array<int64_t, Dims> shape;
  std::array<int64_t, Dims> padded_shape;
  std::array<bool, Dims> flip;
  std::array<int64_t, Dims> permuted_dims;
  size_t normalization_dim = Dims-1;
  size_t normalization_index = 0;
  std::vector<float> mean;
  std::vector<float> inv_stddev;
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SLICE_SLICE_ARGS_H_
