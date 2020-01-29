// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/generic/constant.h"

#include <cuda_runtime.h>
#include <vector>
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

namespace {

template <typename T>
__global__ void Fill(T *data, size_t count, T value) {
  auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    data[i] = value;
}

template <typename Dst, typename Src>
void FillTensorList(
      TensorList<GPUBackend> &dst, const TensorListShape<> &shape, const std::vector<Src> &src,
      cudaStream_t stream) {
  dst.Resize(shape);
  if (src.size() == 1) {
    int64_t size = shape.num_elements();
    int64_t threads = 1024;
    int64_t blocks = div_ceil(size, threads);
    Dst *data = dst.mutable_data<Dst>();
    Fill<<<dim3(blocks), dim3(threads), 0, stream>>>(data, size, ConvertSat<Dst>(src[0]));
  } else {
    SmallVector<Dst, 64> tmp;
    tmp.resize(src.size());
    for (size_t i = 0; i < tmp.size(); i++)
      tmp[i] = ConvertSat<Dst>(src[i]);

    int n = tmp.size() * sizeof(Dst);
    for (int i = 0; i < shape.num_samples(); i++)
      cudaMemcpyAsync(dst.mutable_tensor<Dst>(i), tmp.data(), n, cudaMemcpyHostToDevice, stream);
  }
}

}  // namespace

template <>
void Constant<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  if (output_.ntensor() == 0) {
    TYPE_SWITCH(output_type_, type2id, type, CONSTANT_OP_SUPPORTED_TYPES,
      (
        if (!fdata_.empty()) {
          FillTensorList<type>(output_, output_shape_, fdata_, ws.stream());
        } else {
          assert(!idata_.empty());
          FillTensorList<type>(output_, output_shape_, idata_, ws.stream());
        }
      ), (DALI_FAIL("Unsupported type")));  // NOLINT
  }
  auto &out = ws.OutputRef<GPUBackend>(0);

  out.Reset();
  out.ShareData(&output_);
  out.Resize(output_shape_);
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out.raw_tensor(i) == output_.raw_tensor(i));
  }
  out.SetLayout(layout_);
}

DALI_REGISTER_OPERATOR(Constant, Constant<GPUBackend>, GPU);

}  // namespace dali
