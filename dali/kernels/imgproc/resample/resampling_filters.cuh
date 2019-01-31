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
#include <vector>
#include <memory>
#include <functional>

namespace dali {
namespace kernels {


struct ResamplingFilter {
  float *coeffs;
  int size;
  float scale, anchor;

  __device__ float at(float x) const {
    return at_abs(x*scale + anchor);
  }

  __device__ float at_abs(float x) const {
    if (x < 0)
      return 0;
    if (x > size-1)
      return 0;
    int x0 = x;
    int x1 = min(size-1, x0 + 1);
    float d = x - x0;
    float f0 = __ldg(coeffs + x0);
    float f1 = __ldg(coeffs + x1);
    return f0 + d * (f1 - f0);
  }
};

struct ResamplingFilters {
  std::unique_ptr<float, std::function<void(void*)>> filter_data;

  std::vector<ResamplingFilter> filters;
  ResamplingFilter &operator[](int index) {
    return filters[index];
  }

  const ResamplingFilter &operator[](int index) const {
    return filters[index];
  }
};

std::shared_ptr<ResamplingFilters> GetResamplingFilters(cudaStream_t stream);

}  // namespace kernels
}  // namespace dali
