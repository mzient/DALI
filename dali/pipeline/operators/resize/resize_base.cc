// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/pipeline/operators/resize/resize_base.h"
#include "dali/pipeline/data/views.h"

namespace dali {

inline kernels::ResamplingFilterType interp2resample(DALIInterpType interp) {
#define DALI_MAP_INTERP_TO_RESAMPLE(interp, resample) case DALI_INTERP_##interp:\
  return kernels::ResamplingFilterType::resample;

  switch (interp) {
    DALI_MAP_INTERP_TO_RESAMPLE(NN, Nearest);
    DALI_MAP_INTERP_TO_RESAMPLE(LINEAR, Linear);
    DALI_MAP_INTERP_TO_RESAMPLE(CUBIC, Cubic);
    DALI_MAP_INTERP_TO_RESAMPLE(LANCZOS3, Lanczos3);
    DALI_MAP_INTERP_TO_RESAMPLE(GAUSSIAN, Gaussian);
    DALI_MAP_INTERP_TO_RESAMPLE(TRIANGULAR, Triangular);
  default:
    DALI_FAIL("Unknown interpolation type");
  }
#undef DALI_MAP_INTERP_TO_RESAMPLE
}

void ResizeBase::GetResamplingFilters(kernels::FilterDesc &min_filter,
                                      kernels::FilterDesc &mag_filter,
                                      const OpSpec &spec) {
  DALIInterpType interp_min = DALIInterpType::DALI_INTERP_TRIANGULAR;
  DALIInterpType interp_mag = DALIInterpType::DALI_INTERP_LINEAR;

  if (spec.HasArgument("min_filter"))
    interp_min = spec.GetArgument<DALIInterpType>("min_filter");
  else if (spec.HasArgument("interp_type"))
    interp_min = spec.GetArgument<DALIInterpType>("interp_type");

  if (spec.HasArgument("mag_filter"))
    interp_mag = spec.GetArgument<DALIInterpType>("mag_filter");
  else if (spec.HasArgument("interp_type"))
    interp_mag = spec.GetArgument<DALIInterpType>("interp_type");

  min_filter = { interp2resample(interp_min), 0 };
  mag_filter = { interp2resample(interp_mag), 0 };
}

void ResizeBase::RunGPU(TensorList<GPUBackend> &output,
                        const TensorList<GPUBackend> &input,
                        cudaStream_t stream) {
  output.set_type(input.type());

  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;
  auto &kdata = GetKernelData();

  kdata.context.gpu.stream = stream;
  kdata.requirements = Kernel::GetRequirements(
      kdata.context,
      view<const uint8_t, 3>(input),
      resample_params_);
  kdata.scratch_alloc.Reserve(kdata.requirements.scratch_sizes);

  to_dims_vec(out_shape_, kdata.requirements.output_shapes[0]);
  output.Resize(out_shape_);

  auto scratchpad = kdata.scratch_alloc.GetScratchpad();
  kdata.context.scratchpad = &scratchpad;
  auto in_view = view<const uint8_t, 3>(input);
  auto out_view = view<uint8_t, 3>(output);
  Kernel::Run(kdata.context, out_view, in_view, resample_params_);
}

void ResizeBase::RunCPU(Tensor<CPUBackend> &output,
                        const Tensor<CPUBackend> &input,
                        int thread_idx) {
  using Kernel = kernels::ResampleCPU<uint8_t, uint8_t>;
  auto in_view = view<const uint8_t, 3>(input);
  auto &kdata = GetKernelData(thread_idx);
  kdata.requirements = Kernel::GetRequirements(
      kdata.context,
      in_view,
      resample_params_[thread_idx]);
  kdata.scratch_alloc.Reserve(kdata.requirements.scratch_sizes);
  auto scratchpad = kdata.scratch_alloc.GetScratchpad();
  kdata.context.scratchpad = &scratchpad;

  const auto &input_shape = input.shape();

  auto out_shape = kdata.requirements.output_shapes[0][0];
  out_shape_[thread_idx] = out_shape.shape;

  // Resize the output & run
  output.Resize(out_shape_[thread_idx]);
  auto out_view = view<uint8_t, 3>(output);
  Kernel::Run(kdata.context, out_view, in_view, resample_params_[thread_idx]);
}

}  // namespace dali

