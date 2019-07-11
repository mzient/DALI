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

#ifndef DALI_CORE_CONVERT_H_
#define DALI_CORE_CONVERT_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "dali/core/host_dev.h"
#ifndef __CUDA_ARCH__
#include "dali/util/half.hpp"
#else
#include "dali/core/cuda_utils.h"
#endif
namespace dali {

template <typename T>
struct const_limits;

// std::numeric_limits are not compatible with CUDA
template <typename T>
DALI_HOST_DEV constexpr T max_value() {
  return const_limits<std::remove_cv_t<T>>::max;
}
template <typename T>
DALI_HOST_DEV constexpr T min_value() {
  return const_limits<std::remove_cv_t<T>>::min;
}

#define DEFINE_TYPE_RANGE(type, min_, max_) template <>\
struct const_limits<type> { static constexpr type min = min_, max = max_; }

DEFINE_TYPE_RANGE(bool, false, true);
DEFINE_TYPE_RANGE(uint8_t,  0, 0xff);
DEFINE_TYPE_RANGE(int8_t,  -0x80, 0x7f);
DEFINE_TYPE_RANGE(uint16_t, 0, 0xffff);
DEFINE_TYPE_RANGE(int16_t, -0x8000, 0x7fff);
DEFINE_TYPE_RANGE(uint32_t, 0, 0xffffffff);
DEFINE_TYPE_RANGE(int32_t, -0x80000000, 0x7fffffff);
DEFINE_TYPE_RANGE(uint64_t, 0, 0xffffffffffffffffUL);
DEFINE_TYPE_RANGE(int64_t, -0x8000000000000000L, 0x7fffffffffffffffL);
DEFINE_TYPE_RANGE(float, -3.40282347e+38f, 3.40282347e+38f);
DEFINE_TYPE_RANGE(double, -1.7976931348623157e+308, 1.7976931348623157e+308);

template <typename From, typename To>
struct needs_clamp {
  static constexpr bool from_fp = std::is_floating_point<From>::value;
  static constexpr bool to_fp = std::is_floating_point<To>::value;
  static constexpr bool from_unsigned = std::is_unsigned<From>::value;
  static constexpr bool to_unsigned = std::is_unsigned<To>::value;

  static constexpr bool value =
    // to smaller type of same kind (fp, int)
    (from_fp == to_fp && sizeof(To) < sizeof(From)) ||
    // fp32 has range in excess of (u)int64
    (from_fp && !to_fp) ||
    // converting to unsigned requires clamping negatives to zero
    (!from_unsigned && to_unsigned) ||
    // zero-extending signed unsigned integers requires more bits
    (from_unsigned && !to_unsigned && sizeof(To) <= sizeof(From));
};

template <typename T>
struct ret_type {  // a placeholder for return type
  constexpr ret_type() = default;
};

template <typename T, typename U>
DALI_HOST_DEV constexpr std::enable_if_t<
    needs_clamp<U, T>::value && std::is_signed<U>::value,
    T>
clamp(U value, ret_type<T>) {
  return value <= min_value<T>() ? min_value<T>() :
         value >= max_value<T>() ? max_value<T>() :
         static_cast<T>(value);
}

template <typename T, typename U>
DALI_HOST_DEV constexpr std::enable_if_t<
    needs_clamp<U, T>::value && std::is_unsigned<U>::value,
    T>
clamp(U value, ret_type<T>) {
  return value >= max_value<T>() ? max_value<T>() : static_cast<T>(value);
}

template <typename T, typename U>
DALI_HOST_DEV constexpr std::enable_if_t<
    !needs_clamp<U, T>::value,
    T>
clamp(U value, ret_type<T>) { return value; }

DALI_HOST_DEV constexpr int32_t clamp(uint32_t value, ret_type<int32_t>) {
  return value & 0x80000000u ? 0x7fffffff : value;
}

DALI_HOST_DEV constexpr uint32_t clamp(int32_t value, ret_type<uint32_t>) {
  return value < 0 ? 0u : value;
}

DALI_HOST_DEV constexpr int32_t clamp(int64_t value, ret_type<int32_t>) {
  return value < static_cast<int64_t>(min_value<int32_t>()) ? min_value<int32_t>() :
         value > static_cast<int64_t>(max_value<int32_t>()) ? max_value<int32_t>() :
         static_cast<int32_t>(value);
}

template <>
DALI_HOST_DEV constexpr int32_t clamp(uint64_t value, ret_type<int32_t>) {
  return value > static_cast<uint64_t>(max_value<int32_t>()) ? max_value<int32_t>() :
         static_cast<int32_t>(value);
}

template <>
DALI_HOST_DEV constexpr uint32_t clamp(int64_t value, ret_type<uint32_t>) {
  return value < 0 ? 0 :
         value > static_cast<int64_t>(max_value<uint32_t>()) ? max_value<uint32_t>() :
         static_cast<uint32_t>(value);
}

template <>
DALI_HOST_DEV constexpr uint32_t clamp(uint64_t value, ret_type<uint32_t>) {
  return value > static_cast<uint64_t>(max_value<uint32_t>()) ? max_value<uint32_t>() :
         static_cast<uint32_t>(value);
}

template <typename T>
DALI_HOST_DEV constexpr bool clamp(T value, ret_type<bool>) {
  return static_cast<bool>(value);
}

#ifndef __CUDA_ARCH__
template <typename T>
DALI_HOST_DEV constexpr half_float::half clamp(T value, ret_type<half_float::half>) {
  return static_cast<half_float::half>(value);
}

template <typename T>
DALI_HOST_DEV constexpr T clamp(half_float::half value, ret_type<T>) {
  return clamp(static_cast<float>(value), ret_type<T>());
}

DALI_HOST_DEV inline bool clamp(half_float::half value, ret_type<bool>) {
  return static_cast<bool>(value);
}

DALI_HOST_DEV constexpr half_float::half clamp(half_float::half value,
                                                     ret_type<half_float::half>) {
  return value;
}

#else

template <typename T>
DALI_HOST_DEV constexpr float16 clamp(T value, ret_type<float16>) {
  return static_cast<float16>(value);
}

// __half does not have a constructor for int64_t, use long long
DALI_HOST_DEV inline float16 clamp(int64_t value, ret_type<float16>) {
  return static_cast<float16>(static_cast<long long int>(value));  // NOLINT
}

template <typename T>
DALI_HOST_DEV constexpr T clamp(float16 value, ret_type<T>) {
  return clamp(static_cast<float>(value), ret_type<T>());
}

DALI_HOST_DEV inline bool clamp(float16 value, ret_type<bool>) {
  return static_cast<bool>(value);
}

DALI_HOST_DEV constexpr float16 clamp(float16 value, ret_type<float16>) {
  return value;
}

#endif

template <typename T, typename U>
DALI_HOST_DEV constexpr T clamp(U value) {
  return clamp(value, ret_type<T>());
}

namespace detail {
#ifdef __CUDA_ARCH__

__device__ int cuda_round_helper(float f, int) {  // NOLINT
  return __float2int_rn(f);
}
__device__ unsigned cuda_round_helper(float f, unsigned) {  // NOLINT
  return __float2uint_rn(f);
}
__device__ long long  cuda_round_helper(float f, long long) {  // NOLINT
  return __float2ll_rn(f);
}
__device__ unsigned long long cuda_round_helper(float f, unsigned long long) {  // NOLINT
  return __float2ull_rn(f);
}
__device__ int cuda_round_helper(double f, int) {  // NOLINT
  return __double2int_rn(f);
}
__device__ unsigned cuda_round_helper(double f, unsigned) {  // NOLINT
  return __double2uint_rn(f);
}
__device__ long long  cuda_round_helper(double f, long long) {  // NOLINT
  return __double2ll_rn(f);
}
__device__ unsigned long long cuda_round_helper(double f, unsigned long long) {  // NOLINT
  return __double2ull_rn(f);
}
#endif

template <typename Out, typename In,
  bool OutIsFP = std::is_floating_point<Out>::value,
  bool InIsFP = std::is_floating_point<In>::value>
struct ConverterBase;

template <typename Out, typename In>
struct Converter : ConverterBase<Out, In> {};

/// Converts between two FP types
template <typename Out, typename In>
struct ConverterBase<Out, In, true, true> {
  DALI_HOST_DEV
  static constexpr Out Convert(In value) { return value; }
  DALI_HOST_DEV
  static constexpr Out ConvertNorm(In value) { return value; }
  DALI_HOST_DEV
  static constexpr Out ConvertSat(In value) { return value; }
  DALI_HOST_DEV
  static constexpr Out ConvertSatNorm(In value) { return value; }
};

/// Converts integral to FP type
template <typename Out, typename In>
struct ConverterBase<Out, In, true, false> {
  DALI_HOST_DEV
  static constexpr Out Convert(In value) { return value; }
  DALI_HOST_DEV
  static constexpr Out ConvertSat(In value) { return value; }

  DALI_HOST_DEV
  static constexpr Out ConvertNorm(In value) { return value * (Out(1) / (max_value<In>())); }
  DALI_HOST_DEV
  static constexpr Out ConvertSatNorm(In value) { return value * (Out(1) / (max_value<In>())); }
};

/// Converts FP to integral type
template <typename Out, typename In>
struct ConverterBase<Out, In, false, true> {
  DALI_HOST_DEV
  static constexpr Out Convert(In value) {
#ifdef __CUDA_ARCH__
  return clamp<Out>(detail::cuda_round_helper(value, Out()));
#else
  return clamp<Out>(std::round(value));
#endif
  }

  DALI_HOST_DEV
  static constexpr Out ConvertSat(In value) {
#ifdef __CUDA_ARCH__
  return clamp<Out>(detail::cuda_round_helper(value, Out()));
#else
  return clamp<Out>(std::round(value));
#endif
  }

  DALI_HOST_DEV
  static constexpr Out ConvertNorm(In value) {
#ifdef __CUDA_ARCH__
    return detail::cuda_round_helper(value * max_value<Out>(), Out());
#else
    return std::round(value * max_value<Out>());
#endif
  }

  DALI_HOST_DEV
  static constexpr Out ConvertSatNorm(In value) {
#ifdef __CUDA_ARCH__
    return std::is_signed<Out>::value
      ? clamp<Out>(detail::cuda_round_helper(value * max_value<Out>(), Out()))
      : detail::cuda_round_helper(max_value<Out>() * __saturatef(value), Out());
#else
    return clamp<Out>(std::round(value * max_value<Out>()));
#endif
  }
};

/// Converts signed to signed, unsigned to unsigned or unsigned to signed
template <typename Out, typename In,
          bool IsOutSigned = std::is_signed<Out>::value,
          bool IsInSigned = std::is_signed<In>::value>
struct ConvertIntInt {
  DALI_HOST_DEV
  static constexpr Out Convert(In value) { return value; }
  DALI_HOST_DEV
  static constexpr Out ConvertNorm(In value) {
    return Converter<Out, float>::Convert(value * (1.0f * max_value<Out>() / max_value<In>()));
  }
  DALI_HOST_DEV
  static constexpr Out ConvertSat(In value) { return clamp<Out>(value); }
  DALI_HOST_DEV
  static constexpr Out ConvertSatNorm(In value) {
    return ConvertNorm(value);
  }
};

/// Converts signed to unsigned integer
template <typename Out, typename In>
struct ConvertIntInt<Out, In, false, true> {
  DALI_HOST_DEV
  static constexpr Out Convert(In value) { return value; }
  DALI_HOST_DEV
  static constexpr Out ConvertNorm(In value) {
    return Converter<Out, float>::Convert(value * (1.0f * max_value<Out>() / max_value<In>()));
  }
  DALI_HOST_DEV
  static constexpr Out ConvertSat(In value) { return clamp<Out>(value); }
  DALI_HOST_DEV
  static constexpr Out ConvertSatNorm(In value) {
#ifdef __CUDA_ARCH__
    return detail::cuda_round_helper(
      __saturatef(value * (1.0f / max_value<In>())) * max_value<Out>());
#else
    return value < 0 ? 0 : ConvertNorm(value);
#endif
  }
};

/// Converts between integral types
template <typename Out, typename In>
struct ConverterBase<Out, In, false, false> : ConvertIntInt<Out, In> {
};

/// Pass-through conversion
template <typename T>
struct Converter<T, T> {
  static DALI_HOST_DEV
  constexpr T Convert(T value) { return value; }

  static DALI_HOST_DEV
  constexpr T ConvertSat(T value) { return value; }

    static DALI_HOST_DEV
  constexpr T ConvertNorm(T value) { return value; }

  static DALI_HOST_DEV
  constexpr T ConvertSatNorm(T value) { return value; }
};

template <typename raw_out, typename raw_in>
using converter_t = Converter<
  std::remove_cv_t<raw_out>,
  std::remove_cv_t<std::remove_reference_t<raw_in>>>;;

}  // namespace detail

template <typename Out, typename In>
DALI_HOST_DEV constexpr Out Convert(In value) {
  return detail::converter_t<Out, In>::Convert(value);
}

template <typename Out, typename In>
DALI_HOST_DEV constexpr Out ConvertNorm(In value) {
  return detail::converter_t<Out, In>::ConvertNorm(value);
}

template <typename Out, typename In>
DALI_HOST_DEV constexpr Out ConvertSat(In value) {
  return detail::converter_t<Out, In>::ConvertSat(value);
}

template <typename Out, typename In>
DALI_HOST_DEV constexpr Out ConvertSatNorm(In value) {
  return detail::converter_t<Out, In>::ConvertSatNorm(value);
}

}  // namespace dali

#endif  // DALI_CORE_CONVERT_H_
