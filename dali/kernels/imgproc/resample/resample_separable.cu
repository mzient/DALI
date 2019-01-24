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
#include "../resample.cuh"

namespace dali {
namespace kernels {

struct ResamplingFilters {
  cudaTextureObject_t filterTex;
  cudaArray_t filterData;
};

void InitFilters(ResamplingFilters &filters, cudaStream_t stream) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  const int N = 2;
  const int W = 128;
  cudaMallocArray(&filters.filterData, &desc, W, N);
  cudaMemcpy2DToArrayAsync(filters.filterData, 0, 0, filters.data(), W*sizeof(float), W, N, cudaMemcpyHostToDevice, stream);

  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.sRGB = false;
  texDesc.normalizedCoords = true;
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array = filters.filterData;
  cudaResourceViewDesc viewDesc = {};
  viewDesc.width = W;
  viewDesc.height = H;
  viewDesc.format = cudaResViewFormatFloat1
  cudaCreateTextureObject(&filter.filterTex, &resDesc, &texDesc, &viewDesc);
}

}  // namespace kernels
}  // namespace dali
