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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/common/convert.h"

namespace dali {
namespace kernels {

template <typename Dst, typename Src>
__device__ void ResampleHorz(
    Dst *out, int out_stride, int out_w,
    const Src *in, int in_stride, int in_w, int h, int channels,
    ResamplingFilter filter, int radius) {

  float scale = (float)in_w / out_w;
  float filter_step = filter.scale;

  const int MAX_CHANNELS = 8;

  for (int i = threadIdx.y; i < h; i+=blockDim.y) {
    const Src *in_row = &in[i * in_stride];
    Dst *out_row = &out[i * out_stride];

    for (int j = threadIdx.x; j < out_w; j+=blockDim.x) {
      float xs = (j + 0.5f) * scale;
      int sx0 = xs - radius;
      float f = (sx0 - xs) * filter.scale + filter.anchor;
      float norm = 0;

      float tmp[MAX_CHANNELS];
      for (int c = 0; c < channels; c++)
        tmp[c] = 0;

      for (int k = 0; k < 2*radius+1; k++) {
        int x = sx0 + k;
        int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
        float flt = filter.at_abs(f);
        for (int c = 0; c < channels; c++) {
          Src px = __ldg(in_row + channels * xsample + c);
          tmp[c] += px * flt;
        }
        norm += flt;
        f += filter_step;
      }

      norm = 1.0f/norm;
      for (int c = 0; c < channels; c++)
        out_row[channels * j + c] = clamp<Dst>(tmp[c] * norm);
    }
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
