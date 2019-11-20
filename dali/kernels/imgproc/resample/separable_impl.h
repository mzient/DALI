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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_H_

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/imgproc/resample/resampling_setup.h"
#include "dali/kernels/imgproc/resample/resampling_batch.h"
#include "dali/kernels/common/copy.h"

namespace dali {
namespace kernels {
namespace resampling {

/**
 * @brief Implements a separable resampling filter
 *
 * This implementation can apply differnt resampling filters to each sample.
 * Resampling order is chosen based on input/output shapes and filter type and support.
 * The filter allocates memory only in `Setup` - and even there, it won't reallocate
 * if subsequent calls do not exceed previous number of samples.
 */
template <typename OutputElement, typename InputElement,
          int _spatial_ndim,
          typename Interface = SeparableResamplingFilter<OutputElement, InputElement, _spatial_ndim>
          >
struct SeparableResamplingGPUImpl : Interface {
  using typename Interface::Params;
  using typename Interface::Input;
  using typename Interface::Output;
  using Interface::spatial_ndim;
  using Interface::tensor_ndim;

  using ResamplingSetup = BatchResamplingSetup<spatial_ndim>;
  using SampleDesc = typename ResamplingSetup::SampleDesc;
  using BlockDesc = typename ResamplingSetup::BlockDesc;
  static constexpr int num_tmp_buffers = ResamplingSetup::num_tmp_buffers;
  /**
   * Generates and stores resampling setup
   */
  ResamplingSetup setup;

  using IntermediateElement = float;
  using Intermediate = OutListGPU<IntermediateElement, tensor_ndim>;

  /**
   * The intermediate tensor list
   */
  Intermediate intermediate[num_tmp_buffers];  // NOLINT

  void Initialize(KernelContext &context) {
    setup.Initialize();
  }

  size_t GetTmpMemSize() const {
    // temporary buffers for more than 3D can be reused in a page-flipping fashion
    size_t max[2] = { 0, 0 };
    for (int i = 0; i < setup.num_tmp_buffers; i++) {
      size_t s = setup.intermediate_sizes[i];
      if (s > max[i&1])
        max[i&1] = s;
    }
    return max[0] + max[1];
  }

  size_t GetOddTmpOffset() const {
    // temporary buffers for more than 3D can be reused in a page-flipping fashion
    size_t max_even = 0;
    for (int i = 0; i < setup.num_tmp_buffers; i += 2) {
      size_t s = setup.intermediate_sizes[i];
      if (s > max_even)
        max_even = s;
    }
    return max_even;
  }

  virtual KernelRequirements Setup(KernelContext &context, const Input &in, const Params &params) {
    Initialize(context);
    setup.SetupBatch(in.shape, params);
    // this will allocate and calculate offsets
    for (int i = 0; i < num_tmp_buffers; i++) {
      intermediate[i] = { nullptr, setup.intermediate_shapes[i] };
    }

    KernelRequirements req;
    ScratchpadEstimator se;

    // Sample descriptions need to be delivered to the GPU - hence, the storage
    se.add<SampleDesc>(AllocType::GPU, setup.sample_descs.size());

    // CPU block2sample lookup may change in size and is large enough
    // to mandate declaring it as a requirement for external allocator.
    size_t num_blocks = setup.total_blocks[0] + setup.total_blocks[1];
    se.add<BlockDesc>(AllocType::GPU, num_blocks);
    se.add<BlockDesc>(AllocType::Host, num_blocks);

    // Request memory for intermediate storage.
    se.add<IntermediateElement>(AllocType::GPU, GetTmpMemSize());

    req.scratch_sizes = se.sizes;
    req.output_shapes = { setup.output_shape };
    return req;
  }

  template <typename PassOutputElement, typename PassInputElement>
  void RunPass(
      int which_pass,
      const SampleDesc *descs_gpu,
      const InTensorGPU<BlockDesc, 1> &block2sample,
      cudaStream_t stream) {
    BatchedSeparableResample<spatial_ndim, PassOutputElement, PassInputElement>(
        which_pass,
        descs_gpu, block2sample.data, block2sample.shape[0],
        setup.block_dim,
        stream);
  }

  /**
   * @remarks This function shall not allocate memory by ano other means
   *          than through `context.scratchpad`
   */
  virtual void
  Run(KernelContext &context, const Output &out, const Input &in, const Params &params) {
    cudaStream_t stream = context.gpu.stream;

    SampleDesc *descs_gpu = context.scratchpad->Allocate<SampleDesc>(
        AllocType::GPU, setup.sample_descs.size());

    int total_blocks = setup.total_blocks[0] + setup.total_blocks[1];

    OutTensorCPU<SampleBlockInfo, 1> sample_lookup_cpu = {
      context.scratchpad->Allocate<SampleBlockInfo>(AllocType::Host, total_blocks),
      { total_blocks }
    };
    OutTensorGPU<SampleBlockInfo, 1> sample_lookup_gpu = {
      context.scratchpad->Allocate<SampleBlockInfo>(AllocType::GPU, total_blocks),
      { total_blocks }
    };
    setup.InitializeSampleLookup(sample_lookup_cpu);
    copy(sample_lookup_gpu, sample_lookup_cpu, stream);  // NOLINT (it thinks it's std::copy)


    InTensorGPU<SampleBlockInfo, 1> first_pass_lookup = make_tensor_gpu<1>(
        sample_lookup_gpu.data,
        { setup.total_blocks[0] });

    InTensorGPU<SampleBlockInfo, 1> second_pass_lookup = make_tensor_gpu<1>(
        sample_lookup_gpu.data + setup.total_blocks[0],
        { setup.total_blocks[1] });

    auto *tmp_mem = context.scratchpad->Allocate<IntermediateElement>(
          AllocType::GPU, GetTmpMemSize());

    size_t odd_offset = GetOddTmpOffset();
    for (int t = 0; t < setup.num_tmp_buffers; t++) {
      size_t offset = t&1 ? odd_offset : 0;
      intermediate[t].set_contiguous_data(tmp_mem + offset);
    }

    for (int i = 0; i <in.num_samples(); i++) {
      std::array<IntermediateElement *, num_tmp_buffers> tmp_buffers;
      for (int t = 0; t < setup.num_tmp_buffers; t++)
        tmp_buffers[t] = intermediate[t].tensor_data(i);

      setup.sample_descs[i].set_base_pointers(
        in.tensor_data(i),
        tmp_buffers,
        out.tensor_data(i));
    }

    cudaMemcpyAsync(
        descs_gpu,
        setup.sample_descs.data(),
        setup.sample_descs.size()*sizeof(SampleDesc),
        cudaMemcpyHostToDevice,
        stream);

    RunPass<IntermediateElement, InputElement>(
      0, descs_gpu, first_pass_lookup, stream);
    RunPass<OutputElement, IntermediateElement>(
      1, descs_gpu, second_pass_lookup, stream);
  }
};

}  // namespace resampling
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_H_
