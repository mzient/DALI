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

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_H_

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/convolution/params.h"

namespace dali {
namespace kernels {


template <int channels, typename OutputElement, typename InputElement>
struct SerparableConvoltionFilter {
  using Input = InListGPU<InputElement, 3>;
  using Output = OutListGPU<OutputElement, 3>;

  virtual ~SerparableConvoltionFilter() = default;

  virtual void Setup(KernelContext &context, const Input &in, const SeparableFilterParams &params) = 0;
  virtual void Run(KernelContext &context, const Output &out, const Input &in, const SeparableFilterParams &params) = 0;

  static SerparableConvoltionFilter *CreateImpl(KernelContext &context, const Input &in, const SeparableFilterParams &params);
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_SEPARABLE_H_
