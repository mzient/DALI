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

#include "dali/kernels/slice/slice_flip_normalize_permute_gpu_impl.cuh"

namespace dali {
namespace kernels {


#define INSTANTIATE_SLICE(Out, In, Dims)\
template class SliceFlipNormalizePermuteGPU<Out, In, Dims>;

#define INSTANTIATE_SLICE_OUT(In, Dims)\
INSTANTIATE_SLICE(bool, In, Dims)\
INSTANTIATE_SLICE(uint8_t, In, Dims)\
INSTANTIATE_SLICE(int8_t, In, Dims)\
INSTANTIATE_SLICE(uint16_t, In, Dims)\
INSTANTIATE_SLICE(int16_t, In, Dims)\
INSTANTIATE_SLICE(int32_t, In, Dims)\
INSTANTIATE_SLICE(int64_t, In, Dims)\
INSTANTIATE_SLICE(float, In, Dims)\
INSTANTIATE_SLICE(double, In, Dims)\
INSTANTIATE_SLICE(half, In, Dims)

#define INSTANTIATE_SLICE_IN(Dims)\
INSTANTIATE_SLICE_OUT(bool, Dims)\
INSTANTIATE_SLICE_OUT(uint8_t, Dims)\
INSTANTIATE_SLICE_OUT(int8_t, Dims)\
INSTANTIATE_SLICE_OUT(uint16_t, Dims)\
INSTANTIATE_SLICE_OUT(int16_t, Dims)\
INSTANTIATE_SLICE_OUT(int32_t, Dims)\
INSTANTIATE_SLICE_OUT(int64_t, Dims)\
INSTANTIATE_SLICE_OUT(float, Dims)\
INSTANTIATE_SLICE_OUT(double, Dims)\
INSTANTIATE_SLICE_OUT(half, Dims)

#define INSTANTIATE_SLICE_ALL()\
INSTANTIATE_SLICE_IN(3)\
INSTANTIATE_SLICE_IN(4)

INSTANTIATE_SLICE_ALL()

}  // namespace kernels
}  // namespace dali
