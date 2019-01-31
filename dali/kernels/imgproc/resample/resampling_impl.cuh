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

/// @brief Implements horizontal resampling for a custom ROI
/// @param x0 - start column, in output coordinates
/// @param x1 - end column (exclusive), in output coordinates
/// @param y0 - start row
/// @param y1 - end row (exclusive)
///
/// The function fills the output in block-sized vertical spans.
/// Block horizontal size is warp-aligned.
/// Filter coefficients are pre-calculated for each vertical span to avoid
/// recalculating them for each row, and stored in a shared memory block.
template <typename Dst, typename Src>
__device__ void ResampleHorz(
    int x0, int x1, int y0, int y1,
    float src_x0, float scale,
    Dst *out, int out_stride,
    const Src *in, int in_stride, int in_w, int channels,
    ResamplingFilter filter, int support, int offset) {

  src_x0 += 0.5f * scale - 0.5f;

  const float filter_step = filter.scale;
  const float filter_ofs = filter.anchor + offset * filter_step;

  const int MAX_CHANNELS = 8;
  __shared__ float coeffs[32*256];
  const int coeff_base = support*threadIdx.x;

  for (int j = x0; j < x1; j+=blockDim.x) {
    int dx = j + threadIdx.x;
    const float sx0f = dx * scale + src_x0 + offset;
    const int sx0 = floorf(sx0f);
    float f = (sx0 - sx0f) * filter_step + filter_ofs;
    for (int k = threadIdx.y; k < support; k += blockDim.y) {
      float flt = filter.at_abs(f + k*filter_step);
      coeffs[coeff_base + k] = flt;
    }

    __syncthreads();

    if (dx >= x1)
      break;

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + k];
    }
    norm = 1.0f / norm;

    for (int i = threadIdx.y + y0; i < y1; i+=blockDim.y) {
      const Src *in_row = &in[i * in_stride];
      Dst *out_row = &out[i * out_stride];

      float tmp[MAX_CHANNELS];
      for (int c = 0; c < channels; c++)
        tmp[c] = 0;

      for (int k = 0; k < support; k++) {
        int x = sx0 + k;
        int xsample = x < 0 ? 0 : x >= in_w-1 ? in_w-1 : x;
        float flt = coeffs[coeff_base + k];
        for (int c = 0; c < channels; c++) {
          Src px = __ldg(in_row + channels * xsample + c);
          tmp[c] += px * flt;
        }
        f += filter_step;
      }

      for (int c = 0; c < channels; c++)
        out_row[channels * dx + c] = clamp<Dst>(tmp[c] * norm);
    }
  }
}

/// @brief Implements horizontal resampling for a custom ROI
/// @param x0 - start column, in output coordinates
/// @param x1 - end column (exclusive), in output coordinates
/// @param y0 - start row
/// @param y1 - end row (exclusive)
///
/// The function fills the output in block-sized horizontal spans.
/// Filter coefficients are pre-calculated for each horizontal span to avoid
/// recalculating them for each column, and stored in a shared memory block.
template <typename Dst, typename Src>
__device__ void ResampleVert(
    int x0, int x1, int y0, int y1,
    float src_y0, float scale,
    Dst *out, int out_stride,
    const Src *in, int in_stride, int in_h, int channels,
    ResamplingFilter filter, int support, int offset) {

  src_y0 += 0.5f * scale - 0.5f;

  const float filter_step = filter.scale;
  const float filter_ofs = filter.anchor + offset * filter_step;

  const int MAX_CHANNELS = 8;
  __shared__ float coeffs[32*256];
  const int coeff_base = support*threadIdx.y;

  for (int i = y0; i < y1; i+=blockDim.y) {
    int dy = i + threadIdx.y;
    const float sy0f = dy * scale + src_y0 + offset;
    const int sy0 = floorf(sy0f);
    float f = (sy0 - sy0f) * filter.scale + filter_ofs;
    for (int k = threadIdx.x; k < support; k += blockDim.x) {
      float flt = filter.at_abs(f + k*filter_step);
      coeffs[coeff_base + k] = flt;
    }

    __syncthreads();

    if (dy >= y1)
      break;

    Dst *out_row = &out[dy * out_stride];

    float norm = 0;
    for (int k = 0; k < support; k++) {
      norm += coeffs[coeff_base + k];
    }
    norm = 1.0f / norm;

    for (int j = x0 + threadIdx.x; j < x1; j+=blockDim.x) {
      Dst *out_col = &out_row[j * channels];
      const Src *in_col = &in[j * channels];

      float tmp[MAX_CHANNELS];
      for (int c = 0; c < channels; c++)
        tmp[c] = 0;

      for (int k = 0; k < support; k++) {
        int y = sy0 + k;
        int ysample = y < 0 ? 0 : y >= in_h-1 ? in_h-1 : y;
        float flt = coeffs[coeff_base + k];
        for (int c = 0; c < channels; c++) {
          Src px = __ldg(in_col + in_stride * ysample + c);
          tmp[c] += px * flt;
        }
        f += filter_step;
      }

      for (int c = 0; c < channels; c++)
        out_col[c] = clamp<Dst>(tmp[c] * norm);
    }
  }
}


}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CUH_
