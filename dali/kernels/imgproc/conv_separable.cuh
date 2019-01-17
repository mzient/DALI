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

#include <cuda_runtime.h>
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/imgproc_common.h"

namespace dali {
namespace kernels {

constexpr int AnchorCenter = 0x7fffffff;

enum class BorderMode {
  Clamp, Mirror, Constant
};

template <int channels, typename OutputElement, typename InputElement>
struct SeparableConvolutionGPU {
  using Input = InListGPU<InputElement>;
  using Output = OutListGPU<OutputElement>;

  KernelRequirements GetRequirements(const Input &input, ConvolutionFilter filterHorz, ConvolutionFilter filterVert, BorderMode mode) {
    KernelRequirements req;
    req.output_shapes = { input.shape };
    ScratchpadEstimator se;
    se.add<IntermediateType>(AllocType::GPU, input.shape.total_elements);
    return req;
  }
};

}  // kernels
}  // dali
