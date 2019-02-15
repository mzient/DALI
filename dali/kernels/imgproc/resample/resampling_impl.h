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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_

#include "dali/kernels/static_switch.h"
#include "dali/kernels/common/convert.h"
#include "dali/kernels/imgproc/surface.h"

namespace dali {
namespace kernels {

template <int static_channels, bool clamp_left, bool clamp_right, typename Out, typename In>
void ResampleCol(Out *out, const In *in, int x, int w, const int *in_columns,
                const float *coeffs, int support, int dynamic_channels) {
  const float bias = std::is_integral<Out>::value ? 0.5f : 0;
  const int channels = static_channels < 0 ? dynamic_channels : static_channels;

  int x0 = in_columns[x];
  int k0 = x * support;

  if (static_channels < 0) {
    // we don't know how many channels we have at compile time - inner loop over filter
    for (int c = 0; c < channels; c++) {
      float sum = bias;
      for (int k = 0; k < support; k++) {
        int srcx = x0 + k;
        if (clamp_left) if (srcx < 0) srcx = 0;
        if (clamp_right) if (srcx > w-1) srcx = w-1;
        sum += coeffs[k0 + k] * in[srcx * channels + c];

      }
      out[channels * x + c] = clamp<Out>(sum);
    }
  } else {
    // we know how many channels we have at compile time - inner loop over channels
    float tmp[static_channels > 0 ? static_channels : 1];
    for (int c = 0; c < channels; c++)
      tmp[c] = bias;

    for (int k = 0; k < support; k++) {
      int srcx = x0 + k;
      if (clamp_left) if (srcx < 0) srcx = 0;
      if (clamp_right) if (srcx > w-1) srcx = w-1;
      for (int c = 0; c < channels; c++) {
        tmp[c] += coeffs[k0 + k] * in[srcx * channels + c];
      }
    }

    for (int c = 0; c < channels; c++)
      out[channels * x + c] = clamp<Out>(tmp[c]);
  }
}

template <int static_channels = -1, typename Out, typename In, typename Coeff>
void ResampleHorz_Channels(
    Surface2D<Out> out, Surface2D<const In> in, const int *in_columns,
    const float *coeffs, int support) {
  const int channels = static_channels < 0 ? out.channels : static_channels;

  int first_regular_col = 0;
  int last_regular_col = in.width - 1;
  while (first_regular_col < in.width && in_columns[first_regular_col] < 0)
    first_regular_col++;
  while (last_regular_col >= 0 && in_columns[last_regular_col] + support > in.width)
    last_regular_col--;

  for (int y = 0; y < out.height; y++) {
    Out *out_row = &out(0, y);
    const In *in_row = &in(0, y);

    int x = 0;

    for (; x < first_regular_col && x <= last_regular_col; x++) {
      ResampleCol<static_channels, true, false>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
    for (; x < first_regular_col; x++) {
      ResampleCol<static_channels, true, true>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
    for (; x <= last_regular_col; x++) {
      ResampleCol<static_channels, false, false>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }
    for (; x < out.width; x++) {
      ResampleCol<static_channels, false, true>(
        out_row, in_row, x, in.width, in_columns, coeffs, support, channels);
    }

  }
}


template <typename Out, typename In, typename Coeff>
void ResampleHorz(
    Surface2D<Out> out, Surface2D<const In> in, const int *in_columns,
    const float *col_coeffs, int support) {
  VALUE_SWITCH(out.channels, static_channels, (1, 2, 3, 4), (
    ResampleHorz_Channels<static_channels>(out, in, in_columns, col_coeffs, support);
  ), (
    ResampleHorz_Channels<-1>(out, in, in_columns, col_coeffs, support);
  ));  // NOLINT
}

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_RESAMPLING_IMPL_CPU_H_
