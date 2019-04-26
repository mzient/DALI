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
#include <random>
#include "dali/kernels/tensor_util.h"

namespace dali {
namespace kernels {

TEST(TensorUtil, GatherSamples) {
  int dummy;
  int *base_ptr = &dummy;
  const int ndim = 4;
  std::mt19937_64 rng;
  TensorListView<EmptyBackendTag, int> in;
  TensorListView<EmptyBackendTag, int, ndim> out;
  int num_input_samples = 1000;
  int num_output_samples = 100;
  std::uniform_int_distribution<>
      offset_dist(0, 1000000),
      dim_dist(1, 10),
      idx_dist(0, num_input_samples);
  in.resize(num_input_samples, ndim);

  for (int i = 0; i < num_input_samples; i++) {
    in.data[i] = base_ptr + offset_dist(rng);
    for (int d = 0; d < ndim; d++)
      in.tensor_shape_span(i)[d] = dim_dist(rng);
  }
  std::vector<int> indices(num_output_samples);
  for (auto &idx : indices)
    idx = idx_dist(rng);

  out = gather_samples<ndim>(in, indices);
  ASSERT_EQ(out.num_samples(), static_cast<int>(indices.size()));
  for (int i = 0; i < out.num_samples(); i++) {
    int j = indices[i];
    EXPECT_EQ(out.tensor_shape(i), in.tensor_shape(j));
    EXPECT_EQ(out.tensor_data(i), in.tensor_data(j));
  }
}


TEST(TensorUtil, ScatterSamples) {
  int dummy;
  int *base_ptr = &dummy;
  const int ndim = 4;
  std::mt19937_64 rng;
  TensorListView<EmptyBackendTag, int> in;
  TensorListView<EmptyBackendTag, int, ndim> out;
  int num_input_samples = 100;
  int num_output_samples = 1000;
  std::uniform_int_distribution<>
      offset_dist(0, 1000000),
      dim_dist(1, 10),
      idx_dist(0, num_output_samples);
  in.resize(num_input_samples, ndim);
  out.resize(num_output_samples, ndim);
  std::vector<int> reverse_map(num_output_samples, -1);

  for (int i = 0; i < num_input_samples; i++) {
    in.data[i] = base_ptr + offset_dist(rng);
    for (int d = 0; d < ndim; d++)
      in.tensor_shape_span(i)[d] = dim_dist(rng);
  }
  std::vector<bool> used(num_output_samples);
  std::vector<int> indices(num_input_samples);
  for (int i = 0; i < num_input_samples; i++) {
    int idx;
    do idx = idx_dist(rng); while (used[idx]);
    indices[i] = idx;
    used[idx] = true;
    reverse_map[idx] = i;
  }

  scatter_samples(out, in, indices);
  int checked = 0;
  for (int i = 0; i < out.num_samples(); i++) {
    if (!used[i]) {
      EXPECT_EQ(out.tensor_data(i), nullptr);
    } else {
      int j = reverse_map[i];
      ASSERT_GE(j, 0);
      EXPECT_EQ(out.tensor_shape(i), in.tensor_shape(j));
      EXPECT_EQ(out.tensor_data(i), in.tensor_data(j));
      checked++;
    }
  }
  ASSERT_EQ(checked, num_input_samples) << "Not all input samples used";
}

}  // kernels
}  // dali
