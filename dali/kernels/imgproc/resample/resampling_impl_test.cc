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

#include <gtest/gtest.h>
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"

namespace dali {
namespace kernels {

TEST(ResampleCPU, InitFilter) {
  int w = 10;
  float scale = 1;
  float sx0 = 0;
  auto filter = GaussianFilter(2);
  int support = filter.support();
  std::vector<float> coeffs(w * support);
  std::vector<int> idx(w);
  InitializeFilter(idx.data(), coeffs.data(), w, sx0, scale, filter);
  for (int i = 0; i < w; i++) {
    for (int k = 0; k < support; k++) {
      std::cout << coeffs[i*support + k] << " ";
    }
    std::cout << std::endl;
  }
}

TEST(ResampleCPU, Horizontal) {
//  testing::data::image("imgproc_test/checkerboard.
}

}  // namespace kernels
}  // namespace dali
