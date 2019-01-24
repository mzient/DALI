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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_PARAMS_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_PARAMS_H_

#include <cuda_runtime.h>
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/imgproc_common.h"

namespace dali {
namespace kernels {

enum ResamplingFilterType : uint8_t {
  Gaussian,
  Lanczos,
};

constexpr int KeepOriginalSize = -1;

template <int dim>
struct ResamplingParamsBase {
  ResamplingFilterType type[dim];
  uint8_t radii[dim];
  float dilation[dim];
};

template <int dim>
struct ResamplingParams : ResamplingParamsBase<dim> {
  int output_size[dim];
};

using ResamplingParams2D = ResamplingParams<2>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_PARAMS_H_
