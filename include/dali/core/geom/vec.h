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

#ifndef DALI_CORE_GEOM_VEC_H_
#define DALI_CORE_GEOM_VEC_H_

#include <cmath>
#include "dali/core/host_dev.h"
#include "dali/core/util.h"
#include "dali/core/math_util.h"

namespace dali {

template <size_t N, typename T = float>
struct vec;

template <typename T>
struct is_vec : std::false_type {};

template <size_t N, typename T>
struct is_vec<vec<N, T>> : std::true_type {};

template <size_t N, typename T>
struct vec_base {
  DALI_HOST_DEV
  constexpr vec_base() : v{} {}

  DALI_HOST_DEV
  constexpr vec_base(T scalar) {  // NOLINT
    for (size_t i = 0; i < N; i++)
      v[i] = scalar;
  }

  template <typename... Args, typename =
            std::enable_if_t<dali::all_of<std::is_convertible<Args, T>::value...>::value>>
  DALI_HOST_DEV
  constexpr vec_base(T v0, T v1, Args... args) : v{v0, v1, T(args)... } {}
  T v[N];

  T &operator[](size_t i) { return v[i]; }
  const T &operator[](size_t i) const { return v[i]; }
};

template <typename T>
struct vec_base<1, T> {
  union {
    T v[1];
    T x;
  };

  DALI_HOST_DEV
  constexpr vec_base() : x() {}
  DALI_HOST_DEV
  constexpr vec_base(const T &x) : x(x) {}  // NOLINT
};

template <typename T>
struct vec_base<2, T> {
  union {
    T v[2];
    struct { T x, y; };
  };

  DALI_HOST_DEV
  constexpr vec_base() : x() {}
  DALI_HOST_DEV
  constexpr vec_base(const T &scalar) : x(scalar), y(scalar) {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(const T &x, const T &y) : x(x), y(y) {}
};

template <typename T>
struct vec_base<3, T> {
  union {
    T v[3];
    struct { T x, y, z; };
  };

  DALI_HOST_DEV
  constexpr vec_base() : x() {}
  DALI_HOST_DEV
  constexpr vec_base(const T &scalar) : x(scalar), y(scalar), z(scalar) {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(const T &x, const T &y, const T &z) : x(x), y(y), z(z) {}
};

template <typename T>
struct vec_base<4, T> {
  union {
    T v[4];
    struct { T x, y, z, w; };
  };

  DALI_HOST_DEV
  constexpr vec_base() : x() {}
  DALI_HOST_DEV
  constexpr vec_base(const T &scalar) : x(scalar), y(scalar), z(scalar), w(scalar) {}  // NOLINT
  DALI_HOST_DEV
  constexpr vec_base(const T &x, const T &y, const T &z, const T &w) : x(x), y(y), z(z), w(w) {}
};

template <size_t N, typename T>
struct vec : vec_base<N, T> {
  static_assert(std::is_standard_layout<T>::value,
                "Cannot create a vector ofa non-standard layout type");
  using vec_base<N, T>::vec_base;
  using vec_base<N, T>::v;

  DALI_HOST_DEV
  constexpr T &operator[](size_t i) { return v[i]; }
  DALI_HOST_DEV
  constexpr const T &operator[](size_t i) const { return v[i]; }

  template <typename U>
  DALI_HOST_DEV
  constexpr vec<N, U> cast() const {
    vec<N, U> ret;
    for (size_t i = 0; i < N; i++) {
      ret.v[i] = static_cast<U>(v[i]);
    }
    return ret;
  }

  DALI_HOST_DEV constexpr size_t size() const { return N; }
  DALI_HOST_DEV constexpr vec operator+() const { return *this; }
  DALI_HOST_DEV constexpr T *begin() { return &v[0]; }
  DALI_HOST_DEV constexpr const T *cbegin() const { return &v[0]; }
  DALI_HOST_DEV constexpr const T *begin() const { return &v[0]; }
  DALI_HOST_DEV constexpr T *end() { return &v[N]; }
  DALI_HOST_DEV constexpr const T *cend() const { return &v[N]; }
  DALI_HOST_DEV constexpr const T *end() const { return &v[N]; }

  DALI_HOST_DEV constexpr auto length_square() const {
    decltype(v[0]*v[0] + v[0]*v[0]) ret = v[0]*v[0];
    for (size_t i = 1; i < N; i++)
      ret += v[i]*v[i];
    return ret;
  }
  DALI_HOST_DEV constexpr auto length() const {
#ifdef __CUDA_ARCH__
    return sqrtf(length_square);
#else
  return std::sqrt(length_square());
#endif
  }
  DALI_HOST_DEV constexpr vec normalized() const {
    auto lsq = length_square();
    return *this * rsqrt(lsq);
  }

  DALI_HOST_DEV
  constexpr vec operator-() const {
    vec<N, T> ret;
    for (size_t i = 0; i < N; i++) {
      ret.v[i] = -v[i];
    }
    return ret;
  }
  DALI_HOST_DEV
  constexpr vec operator~() const {
    vec<N, T> ret;
    for (size_t i = 0; i < N; i++) {
      ret.v[i] = ~v[i];
    }
    return ret;
  }

  #define DEFINE_ASSIGN_VEC_OP(op)\
  template <typename U>\
  DALI_HOST_DEV vec &operator op(const vec<N, U> &rhs) {\
    for (size_t i = 0; i < N; i++)\
      v[i] op rhs[i];\
    return *this;\
  }\
  template <typename U>\
  DALI_HOST_DEV std::enable_if_t<!is_vec<U>::value, vec &> operator op(const U &rhs) {\
    for (size_t i = 0; i < N; i++)\
      v[i] op rhs[i];\
    return *this;\
  }

  DEFINE_ASSIGN_VEC_OP(=)
  DEFINE_ASSIGN_VEC_OP(+=)
  DEFINE_ASSIGN_VEC_OP(-=)
  DEFINE_ASSIGN_VEC_OP(*=)
  DEFINE_ASSIGN_VEC_OP(/=)
  DEFINE_ASSIGN_VEC_OP(%=)
  DEFINE_ASSIGN_VEC_OP(&=)
  DEFINE_ASSIGN_VEC_OP(|=)
  DEFINE_ASSIGN_VEC_OP(^=)
  DEFINE_ASSIGN_VEC_OP(<<=)
  DEFINE_ASSIGN_VEC_OP(>>=)
};


template <size_t N, typename T, typename U>
DALI_HOST_DEV
constexpr auto dot(const vec<N, T> &a, const vec<N, U> &b) {
  decltype(a[0]*b[0] + a[0]*b[0]) ret = a[0]*b[0];
  for (size_t i = 1; i < N; i++)
    ret += a[i]*b[i];
  return ret;
}

#define DEFINE_ELEMENTIWSE_VEC_BIN_OP(op)\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV constexpr auto operator op(const vec<N, T> &a, const vec<N, U> &b) {\
  vec<N, decltype(T() op U())> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b[i];\
  return ret;\
}\
template <size_t N, typename T, typename U, typename R = decltype(T() op U())>\
DALI_HOST_DEV constexpr std::enable_if<!is_vec<U>::value, vec<N, R>> \
operator op(const vec<N, T> &a, const U &b) {\
  vec<N, R> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b;\
  return ret;\
}\
template <size_t N, typename T, typename U, typename R = decltype(T() op U())>\
DALI_HOST_DEV constexpr std::enable_if<!is_vec<T>::value, vec<N, R>> \
operator op(const T &a, const vec<N, U> &b) {\
  vec<N, R> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a op b[i];\
  return ret;\
}

DEFINE_ELEMENTIWSE_VEC_BIN_OP(+)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(-)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(*)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(/)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(%)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(&)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(|)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(^)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(<)  // NOLINT
DEFINE_ELEMENTIWSE_VEC_BIN_OP(>)  // NOLINT
DEFINE_ELEMENTIWSE_VEC_BIN_OP(<=)
DEFINE_ELEMENTIWSE_VEC_BIN_OP(>=)

#define DEFINE_SHIFT_VEC_BIN_OP(op)\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV constexpr vec<N, T> operator op(const vec<N, T> &a, const vec<N, U> &b) {\
  vec<N, T> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b[i];\
  return ret;\
}\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV constexpr std::enable_if<!is_vec<U>::value, vec<N, T>> \
operator op(const vec<N, T> &a, const U &b) {\
  vec<N, T> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a[i] op b;\
  return ret;\
}\
template <size_t N, typename T, typename U>\
DALI_HOST_DEV constexpr std::enable_if<!is_vec<T>::value, vec<N, T>> \
operator op(const T &a, const vec<N, U> &b) {\
  vec<N, T> ret;\
  for (size_t i = 0; i < N; i++)\
    ret[i] = a op b[i];\
  return ret;\
}

DEFINE_SHIFT_VEC_BIN_OP(<<)
DEFINE_SHIFT_VEC_BIN_OP(>>)

struct is_true {
  template <typename T>
  DALI_HOST_DEV constexpr bool operator()(const T &x) {
    return static_cast<bool>(x);
  }
};

template <size_t N, typename T, typename Pred = is_true>
DALI_HOST_DEV constexpr bool all(const vec<N, T> &a, Pred P = {}) {
  for (size_t i = 0; i < N; i++)
    if (!P(a[i]))
      return false;
  return true;
}

template <size_t N, typename T, typename Pred = is_true>
DALI_HOST_DEV constexpr bool any(const vec<N, T> &a, Pred P = {}) {
  for (size_t i = 0; i < N; i++)
    if (P(a[i]))
      return true;
  return false;
}

template <size_t N, typename T, typename U>
DALI_HOST_DEV constexpr bool operator==(const vec<N, T> &a, const vec<N, U> &b) {
  for (size_t i = 0; i < N; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

template <size_t N, typename T, typename U>
DALI_HOST_DEV constexpr bool operator!=(const vec<N, T> &a, const vec<N, U> &b) {
  for (size_t i = 0; i < N; i++)
    if (a[i] != b[i])
      return true;
  return false;
}

template <typename U, size_t N, typename T>
DALI_HOST_DEV constexpr auto cast(const vec<N, T> &v) { return v.template cast<U>(); }

template <typename F, size_t N, typename... Elements>
DALI_HOST_DEV constexpr auto elementwise(F f, const vec<N, Elements>&... vecs) {
  using R = decltype(f(vecs[0]...));
  vec<N, R> result;
  for (size_t i = 0; i < N; i++) {
    result[i] = f(vecs[i]...);
  }
  return result;
}

template <size_t N, typename T>
constexpr vec<N, T> clamp(const vec<N, T> &in, const vec<N, T> &lo, const vec<N, T> &hi) {
  return elementwise(clamp, in, lo, hi);
}

template <size_t N, typename T>
constexpr vec<N, T> min(const vec<N, T> &a, const vec<N, T> &b) {
  return elementwise(min, a, b);
}

template <size_t N, typename T>
constexpr vec<N, T> max(const vec<N, T> &a, const vec<N, T> &b) {
  return elementwise(max, a, b);
}

template <size_t N, typename T>
constexpr vec<N, T> floor(const vec<N, T> &a) {
  return elementwise(std::floor, a);
}

template <size_t N, typename T>
constexpr vec<N, T> ceil(const vec<N, T> &a) {
  return elementwise(std::ceil, a);
}

#ifdef __CUDA_ARCH__
template <size_t N>
__device__ constexpr vec<N> floor(const vec<N> &a, const vec<N> &b) {
  return elementwise(floorf, a);
}

template <size_t N>
__device__ constexpr vec<N> ceil(const vec<N> &a, const vec<N> &b) {
  return elementwise(ceilf, a);
}
#endif

template <size_t N>
DALI_HOST_DEV vec<N, int> round_int(const vec<N> &a) {
  return elementwise(static_cast<int(&)(float)>(round_int), a);
}

}  // namespace dali

#endif  // DALI_CORE_GEOM_VEC_H_
