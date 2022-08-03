// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_

#include <cuda_runtime_api.h>
#include "dali/core/geom/vec.h"
#include "dali/core/geom/mat.h"
#include "dali/core/convert.h"

namespace dali {
namespace kernels {
namespace color {

namespace detail {

template <int N, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<N, float> norm(vec<N, Input> x) {
  vec<N, float> out;
  for (int i = 0; i < N; i++)
    out[i] = ConvertNorm<float>(x[i]);
  return out;
}

template <int N>
DALI_HOST_DEV DALI_FORCEINLINE vec<N, float> norm(vec<N, float> x) {
  return x;
}

template <typename From, typename To>
constexpr DALI_HOST_DEV float scale_factor() {
  constexpr double to = is_fp_or_half<To>::value
                      ? 1.0 : static_cast<double>(max_value<To>());

  constexpr double from = is_fp_or_half<From>::value
                        ? 1.0 : static_cast<double>(max_value<From>());

  constexpr float factor = to / from;
  return factor;
}

}  // namespace detail

// Y, Cb, Cr definition from ITU-R BT.601, with values in the range 16-235, allowing for
// footroom and headroom
namespace itu_r_bt_601 {

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_y(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs = vec3(0.257f, 0.504f, 0.098f) * detail::scale_factor<Input, Output>();
  constexpr float bias = 0.0625f * detail::scale_factor<float, Output>();
  float y = dot(coeffs, rgb_in) + bias;
  return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr uint8_t rgb_to_y<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.257f, 0.504f, 0.098f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 16.0f);
}

DALI_HOST_DEV DALI_FORCEINLINE
constexpr uint8_t rgb_to_y(vec<3, uint8_t> rgb) {
  return rgb_to_y<uint8_t, uint8_t>(rgb);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_cb(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs = vec3(-0.148f, -0.291f, 0.439f) * detail::scale_factor<Input, Output>();
  constexpr float bias = 0.5f * detail::scale_factor<float, Output>();
  float y = dot(coeffs, rgb_in) + bias;
  return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cb<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(-0.148f, -0.291f, 0.439f);
  if (rgb.x == rgb.y && rgb.x == rgb.z) return 128;
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cb(vec<3, uint8_t> rgb) {
  return rgb_to_cb<uint8_t, uint8_t>(rgb);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_cr(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs = vec3(0.439f, -0.368f, -0.071f) * detail::scale_factor<Input, Output>();
  constexpr float bias = 0.5f * detail::scale_factor<float, Output>();
  float y = dot(coeffs, rgb_in) + bias;
  return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cr<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.439f, -0.368f, -0.071f);
  if (rgb.x == rgb.y && rgb.x == rgb.z) return 128;
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128.0f);
}

DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cr(vec<3, uint8_t> rgb) {
  return rgb_to_cr<uint8_t, uint8_t>(rgb);
}


// Gray uses the full dynamic range of the type (e.g. 0..255)
// while ITU-R BT.601 uses a reduced range to allow for floorroom and footroom (e.g. 16..235)
template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output y_to_gray(Input gray_in) {
  auto gray = detail::norm(gray_in);  // TODO(janton): optimize number of multiplications
  constexpr float scale = 1 / (0.257f +  0.504f + 0.098f);
  return ConvertSatNorm<Output>(scale * (gray - 0.0625f));
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t y_to_gray<uint8_t, uint8_t>(uint8_t gray) {
  constexpr float scale = 1 / (0.257f + 0.504f + 0.098f);
  return ConvertSat<uint8_t>(scale * (gray - 16));
}

DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t y_to_gray(uint8_t gray) {
  return y_to_gray<uint8_t, uint8_t>(gray);
}


template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output gray_to_y(Input y) {
  constexpr float scale = (0.257f + 0.504f + 0.098f) * detail::scale_factor<Input, Output>();
  constexpr float bias = 0.0625f * detail::scale_factor<float, Output>();
  return ConvertSat<Output>(y * scale + bias);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr uint8_t gray_to_y<uint8_t, uint8_t>(uint8_t y) {
  constexpr float scale = 0.257f + 0.504f + 0.098f;
  return ConvertSat<uint8_t>(y * scale + 0.0625f);
}

DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t gray_to_y(uint8_t y) {
  return gray_to_y<uint8_t, uint8_t>(y);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr_in) {
  auto ycbcr = detail::norm(ycbcr_in);  // TODO(janton): optimize number of multiplications
  auto tmp_y = 1.164f * (ycbcr.x - 0.0625f);
  auto tmp_b = ycbcr.y - 0.5f;
  auto tmp_r = ycbcr.z - 0.5f;
  auto r = ConvertSatNorm<Output>(tmp_y + 1.596f * tmp_r);
  auto g = ConvertSatNorm<Output>(tmp_y - 0.813f * tmp_r - 0.392f * tmp_b);
  auto b = ConvertSatNorm<Output>(tmp_y + 2.017f * tmp_b);
  return {r, g, b};
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr vec<3, uint8_t> ycbcr_to_rgb<uint8_t, uint8_t>(vec<3, uint8_t> ycbcr) {
  auto tmp_y = 1.164f * (ycbcr.x - 16);
  auto tmp_b = ycbcr.y - 128;
  auto tmp_r = ycbcr.z - 128;
  auto r = ConvertSat<uint8_t>(tmp_y + 1.596f * tmp_r);
  auto g = ConvertSat<uint8_t>(tmp_y - 0.813f * tmp_r - 0.392f * tmp_b);
  auto b = ConvertSat<uint8_t>(tmp_y + 2.017f * tmp_b);
  return {r, g, b};
}

}  // namespace itu_r_bt_601

// Y, Cb, Cr formulas used in JPEG, using the whole dynamic range of the type.
namespace jpeg {

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_y(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs = vec3(0.299f, 0.587f, 0.114f) * detail::scale_factor<Input, Output>();
  float y = dot(coeffs, rgb_in);
  return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE
constexpr uint8_t rgb_to_y<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.299f, 0.587f, 0.114f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb));
}

DALI_HOST_DEV DALI_FORCEINLINE
constexpr uint8_t rgb_to_y(vec<3, uint8_t> rgb) {
  return rgb_to_y<uint8_t, uint8_t>(rgb);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_cb(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs = vec3(-0.16873589f, -0.33126411f, 0.5f) * detail::scale_factor<Input, Output>();
  constexpr float bias = 0.5f * detail::scale_factor<float, Output>();
  float y = dot(coeffs, rgb_in) + bias;
  return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cb<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(-0.16873589f, -0.33126411f, 0.5f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128);
}

DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cb(vec<3, uint8_t> rgb) {
  return rgb_to_cb<uint8_t, uint8_t>(rgb);
}

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE constexpr Output rgb_to_cr(vec<3, Input> rgb_in) {
  constexpr vec3 coeffs = vec3(0.5f, -0.41868759f, -0.08131241f) * detail::scale_factor<Input, Output>();
  constexpr float bias = 0.5f * detail::scale_factor<float, Output>();
  float y = dot(coeffs, rgb_in) + bias;
  return needs_clamp<Input, Output>::value ? Convert<Output>(y) : ConvertSat<Output>(y);
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cr<uint8_t, uint8_t>(vec<3, uint8_t> rgb) {
  constexpr vec3 coeffs(0.5f, -0.41868759f, -0.08131241f);
  return ConvertSat<uint8_t>(dot(coeffs, rgb) + 128);
}

DALI_HOST_DEV DALI_FORCEINLINE constexpr uint8_t rgb_to_cr(vec<3, uint8_t> rgb) {
  return rgb_to_cr<uint8_t, uint8_t>(rgb);
}

///////////////////////////////

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, Output> ycbcr_to_rgb(vec<3, Input> ycbcr_in) {
  auto ycbcr = detail::norm(ycbcr_in);  // TODO(janton): optimize number of multiplications
  float tmp_b = ycbcr.y - 0.5f;
  float tmp_r = ycbcr.z - 0.5f;
  auto r = ConvertSatNorm<Output>(ycbcr.x + 1.402f * tmp_r);
  auto g = ConvertSatNorm<Output>(ycbcr.x - 0.34413629f * tmp_b - 0.71413629f * tmp_r);
  auto b = ConvertSatNorm<Output>(ycbcr.x + 1.772f * tmp_b);
  return {r, g, b};
}

template <>
DALI_HOST_DEV DALI_FORCEINLINE vec<3, uint8_t> ycbcr_to_rgb(vec<3, uint8_t> ycbcr) {
  float tmp_b = ycbcr.y - 128;
  float tmp_r = ycbcr.z - 128;
  auto r = ConvertSat<uint8_t>(ycbcr.x + 1.402f * tmp_r);
  auto g = ConvertSat<uint8_t>(ycbcr.x - 0.34413629f * tmp_b - 0.71413629f * tmp_r);
  auto b = ConvertSat<uint8_t>(ycbcr.x + 1.772f * tmp_b);
  return {r, g, b};
}


}  // namespace jpeg

template <typename Output, typename Input>
DALI_HOST_DEV DALI_FORCEINLINE Output rgb_to_gray(vec<3, Input> rgb) {
  return jpeg::rgb_to_y<Output>(rgb);
}

}  // namespace color
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_SPACE_CONVERSION_IMPL_H_
