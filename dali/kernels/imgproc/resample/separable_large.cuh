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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_LARGE_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_LARGE_CUH_

#include "dali/kernels/imgproc/resample/separable.h"

namespace dali {
namespace kernels {

template <int channels, typename OutputElement, typename InputElement,
          typename Base_ = SerparableResamplingFilter<channels, OutputElement, InputElement>>
struct LargeSeparableResamplingGPU : Base_ {
  using Base = Base_;
  using typename Base::Input;
  using typename Base::Output;

  static TensorListShape<2> IntermediateImageShapes(const Input &in, const SeparableFilterParams &params) {
    TensorListShape<2> sh;
    sh.resize(in.num_samples());

    int dw = 0, dh = 0;
    if (params.filterHorz.size() < params.filterVert.size())
      dw = params.filterHorz.size();
    else
      dh = params.filterVert.size();

    for (int i = 0; i < sh.num_samples(); i++) {
      auto ts_out = sh.tensor_shape_span(i);
      auto ts_in = in.shape.tensor_shape_span(i);
      ts_out[0] = ts_in[0] + dh;
      ts_out[1] = ts_in[1] + dw;
    }
    return sh;
  }

  virtual KernelRequirements Setup(KernelContext &context, const Input &in, const SeparableFilterParams &params) {
    KernelRequirements req;
    req.output_shapes = { input.shape };
  }

  virtual void Run(KernelContext &context, const Output &out, const Input &in, const SeparableFilterParams &params) {
    // TODO
  }
};

}  // namespace kerenls
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_LARGE_CUH_
