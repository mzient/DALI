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

#ifndef DALI_KERNELS_TENSOR_UTIL_H_
#define DALI_KERNELS_TENSOR_UTIL_H_

#include <cassert>
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/range_check.h"

namespace dali {
namespace kernels {

template <typename T, typename U = remove_cv_ref_t<T>>
struct is_tensor_view : std::false_type {};

template <typename T, typename Backend, typename DataType, int ndim>
struct is_tensor_view<T, TensorView<Backend, DataType, ndim>> : std::true_type {};

template <typename T, typename U = remove_cv_ref_t<T>>
struct is_tensor_list_view : std::false_type {};

template <typename T, typename Backend, typename DataType, int ndim>
struct is_tensor_list_view<T, TensorListView<Backend, DataType, ndim>> : std::true_type {};

template <typename Backend1, typename Data1, int ndim1,
          typename Backend2, typename Data2, int ndim2>
void check_sample_dim(const TensorListView<Backend1, Data1, ndim1> &list1,
                      const TensorListView<Backend2, Data2, ndim2> &list2) {
    detail::check_compatible_ndim<ndim1, ndim2>();
    if (list1.sample_dim() != list1.sample_dim())
      throw std::logic_error("Incompatible number of dimensions");
}

template <typename Backend, typename U, typename T, int out_ndim, int in_ndim>
void copy_sample(TensorListView<Backend, U, out_ndim> &out, int out_idx,
                 const TensorListView<Backend, T, in_ndim> &in, int in_idx) {
  detail::check_compatible_ndim<out_ndim, in_ndim>();
  int sample_dim = in.sample_dim();
  for (int d = 0; d < sample_dim; d++)
    out.tensor_shape_span(out_idx)[d] = in.tensor_shape_span(in_idx)[d];
  out.data[out_idx] = in.data[in_idx];
}

template <typename Backend, typename OutData, int out_ndim,
          typename InData, int in_ndim, typename IndexCollection>
void gather_samples(
      TensorListView<Backend, OutData, out_ndim> &out,
      const TensorListView<Backend, InData, in_ndim> &in,
      const IndexCollection &indices) {
  detail::check_compatible_ndim<in_ndim, out_ndim>();
  int sample_dim = in.sample_dim();
  out.resize(size(indices), sample_dim);
  int i = 0;
  for (int index : indices) {
    assert(index >= 0 && index < in.num_samples());
    copy_sample(out, i++, in, index);
  }
}

template <int ndim = InferDimensions,
          typename Backend,
          typename DataType,
          int in_ndim,
          typename IndexCollection,
          int out_ndim = INFER_OUT_DIM(ndim, in_ndim)>
TensorListView<Backend, DataType, out_ndim> gather_samples(
      const TensorListView<Backend, DataType, in_ndim> &in,
      const IndexCollection &indices) {
  TensorListView<Backend, DataType, out_ndim> out;
  gather_samples(out, in, indices);
  return out;
}

template <typename Backend, typename OutData, int out_ndim,
          typename InData, int in_ndim, typename IndexCollection>
void scatter_samples(
      TensorListView<Backend, OutData, out_ndim> &out,
      const TensorListView<Backend, InData, in_ndim> &in,
      const IndexCollection &indices) {
  check_sample_dim(out, in);
  int sample_dim = in.sample_dim();
  assert(indices.size() <= in.num_samples());
  int i = 0;
  for (int index : indices) {
    assert(index >= 0 && index < out.num_samples());
    copy_sample(out, index, in, i++);
  }
}


/// @brief Fills some samples in the output with corresponding samples from the input
/// @param out            output tensor list
/// @param in             input tensor list
/// @param index_out_in   a collection of pairs (output_index, input_index);
///        its elements must have fields `first` and `second`, implicitly convertible to `int'
///
/// The operation performed is:
/// ```
/// for pair in index_out_in
///     out[pair.second] = in[pair.first]
/// ```
template <typename Backend, typename OutData, int out_ndim,
          typename InData, int in_ndim, typename IndexPairCollection>
void insert_samples(
      TensorListView<Backend, OutData, out_ndim> &out,
      const TensorListView<Backend, InData, in_ndim> &in,
      const IndexPairCollection &index_out_in) {
  int sample_dim = in.sample_dim();
  check_sample_dim(out, in);
  range_check(size(index_out_in), 0, in.num_samples()+1);
  int i = 0;
  for (auto index_pair : index_out_in) {
    int out_idx = index_pair.first;
    int in_idx = index_pair.second;
    range_check(out_idx, 0, out.num_samples());
    range_check(in_idx, 0, in.num_samples());
    copy_sample(out, out_idx, in, in_idx);
  }
}


/// @brief Fills output with samples selected from multiple inputs
/// @param out        output tensor list
/// @param in_lists   an indexable collection of tensor lists
/// @param list_and_sample_indices  a collection of pairs (list_index, sample_index);
///        its elements must have fields `first` and `second`, implicitly convertible to `int'
template <typename Backend, typename OutData, int out_ndim,
          typename InLists, typename IndexPairCollection>
typename std::enable_if<is_tensor_list_view<decltype(std::declval<InLists>()[0])>::value>::type
gather_samples(
      TensorListView<Backend, OutData, out_ndim> &out,
      const InLists &in_lists,
      const IndexPairCollection &list_and_sample_indices) {
  const int in_ndim = remove_ref_t<decltype(in_lists[0])>::compile_time_sample_dim;
  detail::check_compatible_ndim<in_ndim, out_ndim>();
  if (size(list_and_sample_indices) == 0) {
    out.resize(0);
    return;
  }
  int num_lists = size(in_lists);
  assert(num_lists > 0);
  int sample_dim = in_lists[0].sample_dim();
  int num_samples = list_and_sample_indices.size();
  out.resize(num_samples, sample_dim);
  int i = 0;
  for (auto index_pair : list_and_sample_indices) {
    int list_idx = index_pair.first;
    int sample_idx = index_pair.first;
    range_check(list_idx, 0, num_lists);
    auto &list = in_lists[list_idx];
    check_sample_dim(out, list);
    range_check(sample_idx, 0, list.num_samples());
    copy_sample(out, i++, list, sample_idx);
  }
}


/// @brief Fills output with samples selected from multiple inputs
/// @param out        output tensor list
/// @param in_lists   an indexable collection of tensor lists
/// @param list_and_sample_indices  a collection of pairs (list_index, sample_index);
///        its elements must have fields `first` and `second`, implicitly convertible to `int'
template <typename Backend, typename OutLists,
          typename InData, int in_ndim, typename IndexPairCollection>
typename std::enable_if<is_tensor_list_view<decltype(std::declval<OutLists>()[0])>::value>::type
scatter_samples(
      const OutLists &out_lists,
      const TensorListView<Backend, InData, in_ndim> &in,
      const IndexPairCollection &list_and_sample_indices) {
  const int out_ndim = remove_ref_t<decltype(out_lists[0])>::compile_time_sample_dim;
  detail::check_compatible_ndim<in_ndim, out_ndim>();
  range_check(size(list_and_sample_indices), 0, in.num_samples() + 1);
  int num_lists = size(out_lists);
  int num_samples = in.num_samples();
  int i = 0;
  for (auto index_pair : list_and_sample_indices) {
    int list_idx = index_pair.first;
    int sample_idx = index_pair.first;
    range_check(list_idx, 0, num_lists);
    auto &list = out_lists[list_idx];
    check_sample_dim(list, in);
    range_check(sample_idx, 0, list.num_samples());
    copy_sample(list, sample_idx, in, i++);
  }
}

template <typename Backend,
          typename DataOut,
          int out_ndim,
          typename DataIn,
          int in_ndim,
          typename IterableBitMask>
void mask_gather(
      TensorListView<Backend, DataOut, out_ndim> &out,
      const TensorListView<Backend, DataIn, in_ndim> &in,
      const IterableBitMask &mask) {
  detail::check_compatible_ndim<in_ndim, out_ndim>();
  int sample_dim = in.sample_dim();
  int nonzero = 0;
  for (auto m : mask) {
    if (m) nonzero++;
  }
  out.resize(nonzero, sample_dim);
  int i = 0, j = 0;
  for (auto m : mask) {
    if (m)
      copy_sample(out, j++, in, i);
    i++;
  }
}

template <int ndim = InferDimensions,
          typename Backend,
          typename DataType,
          int in_ndim,
          typename IterableBitMask,
          int out_ndim = INFER_OUT_DIM(ndim, in_ndim)>
TensorListView<Backend, DataType, out_ndim> mask_gather(
      const TensorListView<Backend, DataType, ndim> &in,
      const IterableBitMask &mask) {
  TensorListView<Backend, DataType, out_ndim> out;
  mask_gather(out, in, mask);
  return out;
}

template <typename Backend,
          typename DataOut,
          int out_ndim,
          typename DataIn,
          int in_ndim,
          typename IterableBitMask>
void mask_insert(
      TensorListView<Backend, DataOut, out_ndim> &out,
      const TensorListView<Backend, DataIn, in_ndim> &in,
      const IterableBitMask &mask) {
  check_sample_dim(out, in);
  int sample_dim = in.sample_dim();
  int in_idx = 0;
  int out_idx = 0;
  for (auto m : mask) {
    if (m) {
      check_range(in_idx, 0, in.num_samples());
      check_range(out_idx, 0, out.num_samples());
      copy_sample(out, out_idx, in, in_idx);
      in_idx++;
    }
    out_idx++;
  }
}


template <typename Backend, typename OutData, int out_ndim, typename InLists>
void concat(
      TensorListView<Backend, OutData, out_ndim> &out,
      const InLists &in_lists) {
  const int in_ndim = remove_ref_t<decltype(in_lists[0])>::compile_time_sample_dim;
  detail::check_compatible_ndim<in_ndim, out_ndim>();

  int sample_dim = -1;
  int total_samples = 0;
  for (auto &list : in_lists) {
    if (list.empty())
      continue;
    int d = list.sample_dim();
    if (sample_dim < 0)
      sample_dim = d;
    else if (sample_dim != d)
      throw std::logic_error("Input contains lists of different dimensionality");
    total_samples += list.num_samples();
  }

  if (total_samples == 0) {
    out.resize(0);
    return;
  }

  int i = 0;

  for (auto &list : in_lists) {
    if (list.empty())
      continue;
    int n = list.num_samples();
    for (int j = 0; j < n; j++)
      copy_sample(out, i+j, list, j);
    i += n;
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TENSOR_UTIL_H_
