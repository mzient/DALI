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

#include "dali/kernels/signal/window/extract_windows_gpu.h"
#include "dali/kernels/signal/window/extract_windows_gpu.cuh"

namespace dali {
namespace kernels {
namespace signal {

template <typename Dst, typename Src>
struct ExtractWindowsGpu<Dst, Src>::Impl : public ExtractWindowsGpuImpl<Dst, Src> {
};

template <typename Dst, typename Src>
KernelRequirements ExtractWindowsGpu<Dst, Src>::Setup(
    KernelContext &context,
    const InListGPU<Src, 1> &input,
    const InTensorGPU<float, 1> &window,
    const ExtractWindowsBatchedArgs &args) {
  if (!impl)
    impl = std::make_unique<Impl>();

  return impl->Setup(context, input, args, args.concatenate, args.padded_output_window);
}

template <typename Dst, typename Src>
void ExtractWindowsGpu<Dst, Src>::Run(
    KernelContext &context,
    const OutListGPU<Dst, 2> &output,
    const InListGPU<Src, 1> &input,
    const InTensorGPU<float, 1> &window,
    const ExtractWindowsBatchedArgs &args) {
  (void)args;
  assert(impl != nullptr);
  impl->Run(context, output, input, window);
}

template <typename Dst, typename Src>
ExtractWindowsGpu<Dst, Src>::ExtractWindowsGpu() = default;

template <typename Dst, typename Src>
ExtractWindowsGpu<Dst, Src>::~ExtractWindowsGpu() = default;

template class ExtractWindowsGpu<float, float>;
template class ExtractWindowsGpu<float, int16_t>;
template class ExtractWindowsGpu<float, int8_t>;

}  // namespace signal
}  // namespace kernels
}  // namespace dali
