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
  None = -1, Clamp = 0, Mirror, Constant
};

struct SeparableFilterParams {
  ConvolutionFilter filterHorz;
  ConvolutionFilter filterVert;
  BorderMode mode;
};

template <int channels, typename OutputElement, typename InputElement>
struct LargeSeparableConvolutionGPU {
  using Input = InListGPU<InputElement>;
  using Output = OutListGPU<OutputElement>;

  static TensorListShape<2> IntermediateImageShapes(const Input &in, const SeparableFilterParams &params) {
    TensorListShape<2> sh;
    sh.resize(in.num_samples());

    int dw = 0, dh = 0;
    if (params.filterHorz.size() < params.filterVert.size())
      dw = params.filterHorz.size();
    else
      dh = params.filterVert.size();

    for (int i = 0; i < sh.num_samples(); i++) {
      auto ts = sh.tensor_shape_span(u);
      ts[0] += dh;
      ts[1] += dw;
    }
    return ts;
  }

  void Setup()

};


template <int channels, typename OutputElement, typename InputElement>
struct SeparableConvolutionGPU {
  using Input = InListGPU<InputElement>;
  using Output = OutListGPU<OutputElement>;

  KernelRequirements GetRequirements(KernelContext &contex, const Input &input, const SeparableFilterParams &params) {
    KernelRequirements req;
    req.output_shapes = { input.shape };
    ScratchpadEstimator se;
    req.scratch_sizes = se.sizes;
    return req;
  }

  void Run(KernelContext &contex, const Output &output, const Input &input, const SeparableFilterParams &params) {

  }
};

}  // kernels
}  // dali
