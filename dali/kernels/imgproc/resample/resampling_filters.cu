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
#include <mutex>
#include <unordered_map>
#include "dali/kernels/imgproc/resample.cuh"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/alloc.h"
#include "dali/kernels/span.h"

namespace dali {
namespace kernels {

template <typename Function>
__device__ inline void InitFilter(ResamplingFilter &filter, Function F) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i > filter.size)
    return;
  float x = (i - filter.anchor) * filter.scale;
  filter.coeffs[i] = F(x);
}

__global__ void InitGaussianFilter(ResamplingFilter filter) {
  InitFilter(filter, [](float x) {
    return expf(-x*x);
  });
}

inline __host__ __device__ float sinc(float x) {
  return x ? sinf(x) / x : 0;
}

__global__ void InitLanczosFilter(ResamplingFilter filter, float a) {
  InitFilter(filter, [a](float x) {
    if (fabsf(x) >= a)
      return 0.0f;
    return sinc(x)*sinc(x / a);
  });
}

void InitFilters(ResamplingFilters &filters, cudaStream_t stream) {
  int lanczos_resolution = 8;
  int lanczos_a = 3;
  int triangular_size = 3;
  int gaussian_size = 31;
  int lanczos_size = (2*lanczos_a*lanczos_resolution + 1);
  int total_size = triangular_size + gaussian_size + lanczos_size;

  filters.filter_data = memory::alloc_unique<float>(AllocType::GPU, total_size);

  auto add_filter = [&](int size, float scale) {
    float *base = filters.filters.empty()
        ? filters.filter_data.get()
        : filters.filters.back().coeffs + filters.filters.back().size;
    filters.filters.push_back({ base, size, scale, (size - 1) * 0.5f});
  };
  add_filter(triangular_size, 1.0f);
  add_filter(gaussian_size, 0.25f / gaussian_size);
  add_filter(lanczos_size, 1.0f / lanczos_resolution);

  float triangle[3] = { 0, 1, 0 };

  cudaMemcpyAsync(filters.filters[0].coeffs, triangle,
                  3*sizeof(float), cudaMemcpyHostToDevice, stream);

  InitGaussianFilter<<<1, gaussian_size, 0, stream>>>(filters.filters[1]);
  InitLanczosFilter<<<1, lanczos_size, 0, stream>>>(filters.filters[2], lanczos_a);
}

static std::unordered_map<int, std::weak_ptr<ResamplingFilters>> filters;
static std::mutex filter_mutex;

std::shared_ptr<ResamplingFilters> GetResamplingFilters(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(filter_mutex);
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return nullptr;

  auto ptr = filters[device].lock();
  if (!ptr) {
    ptr = std::make_shared<ResamplingFilters>();
    InitFilters(*ptr, stream);
    filters[device] = ptr;
  }
  return ptr;
}

}  // namespace kernels
}  // namespace dali
