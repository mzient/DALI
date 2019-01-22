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

#ifndef DALI_KERNELS_IMGPROC_CONV_SEPARABLE_CUH_
#define DALI_KERNELS_IMGPROC_CONV_SEPARABLE_CUH_

#include <cuda_runtime.h>
#include <memory>
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/imgproc_common.h"
#include "dali/kernels/imgproc/convolution/params.h"
#include "dali/kernels/imgproc/convolution/separable.h"

namespace dali {
namespace kernels {

template <int channels, typename OutputElement, typename InputElement,
          typename Base_ = SerparableConvoltionFilter<channels, OutputElement, InputElement>>
struct LargeSeparableConvolutionGPU : Base_ {
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
      auto ts = sh.tensor_shape_span(u);
      ts[0] += dh;
      ts[1] += dw;
    }
    return ts;
  }

  virtual KernelRequirements Setup(KernelContext &context, const Input &in, const SeparableFilterParams &params) {
  }

  virtual void Run(KernelContext &context, const Output &out, const Input &in, const SeparableFilterParams &params) {
    // TODO
  }
};

template <int channels, typename OutputElement, typename InputElement>
struct SeparableConvolutionGPU {
  using Input = InListGPU<InputElement, 3>;
  using Output = OutListGPU<OutputElement, 3>;

  using Impl = SerparableConvoltionFilter<channels, OutputElement, InputElement>;
  using ImplPtr = std::unique_ptr<Impl>;

  static ImplPtr SelectImpl(
      KernelContext &context,
      const Input &input,
      const SeparableFilterParams &params) {
    auto cur_impl = any_cast<ImplPtr*>(context.kernel_data);
    if (cur_impl)
      return std::move(*cur_impl);
    else
      return ImplPtr(new Impl(params));
  }

  static KernelRequirements GetRequirements(KernelContext &contex, const Input &input, const SeparableFilterParams &params) {
    req.output_shapes = { input.shape };
    auto impl = SelectImpl(context, input, params);
    auto req = impl->Setup();
    context.data = std::move(impl);
    return req;
  }

  static void Run(KernelContext &contex, const Output &output, const Input &input, const SeparableFilterParams &params) {

  }
};

}  // kernels
}  // dali

#endif  // DALI_KERNELS_IMGPROC_CONV_SEPARABLE_CUH_
