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
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/imgproc/resample.h"

using std::cout;
using std::endl;

namespace dali {
namespace kernels {

namespace {
void RandomParams(
    TensorListShape<3> &tls,
    std::vector<ResamplingParams2D> &params,
    int num_samples) {
  std::mt19937_64 rng;
  auto size_dist    = uniform_distribution(128, 2048);
  auto channel_dist = uniform_distribution(1, 4);
  auto aspect_dist  = uniform_distribution(0.5f, 2.0f);

  tls.resize(num_samples);
  params.resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    auto ts = tls.tensor_shape_span(i);
    float aspect = sqrt(aspect_dist(rng));
    ts[0] = size_dist(rng) * aspect;
    ts[1] = size_dist(rng) / aspect;
    ts[2] = channel_dist(rng);
    aspect = sqrt(aspect_dist(rng));
    params[i][0].output_size = size_dist(rng) * aspect;
    params[i][1].output_size = size_dist(rng) / aspect;
    params[i][0].min_filter.type = ResamplingFilterType::Triangular;
    params[i][0].mag_filter.type = ResamplingFilterType::Linear;
    params[i][1].min_filter.type = ResamplingFilterType::Triangular;
    params[i][1].mag_filter.type = ResamplingFilterType::Linear;
  }
}
}  // namespace

TEST(SeparableImpl, Setup) {
  int N = 32;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  SeparableResamplingGPUImpl<uint8_t, uint8_t> resampling;
  TestTensorList<uint8_t, 3> input, output;

  TensorListShape<3> tls;
  std::vector<ResamplingParams2D> params;
  RandomParams(tls, params, N);

  input.reshape(tls);

  InListGPU<uint8_t, 3> in_tv = input.gpu();

  auto req = resampling.Setup(ctx, in_tv, params);
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tv = output.gpu();

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      tls.tensor_shape_span(i)[2]
    };
    EXPECT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);

    EXPECT_EQ(in_tv.offsets[i], resampling.setup.sample_descs[i].offsets[0]);
    EXPECT_EQ(resampling.intermediate.offsets[i], resampling.setup.sample_descs[i].offsets[1]);
    EXPECT_EQ(out_tv.offsets[i], resampling.setup.sample_descs[i].offsets[2]);

    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[0], 1);
    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[1], 1);
  }
  EXPECT_GT(resampling.setup.total_blocks.pass[0], N);
  EXPECT_GT(resampling.setup.total_blocks.pass[1], N);
}

constexpr FilterDesc tri(float radius = 0) {
  return { ResamplingFilterType::Triangular, radius };
}

constexpr FilterDesc lin() {
  return { ResamplingFilterType::Linear, 0 };
}

constexpr FilterDesc lanczos() {
  return { ResamplingFilterType::Lanczos3, 0 };
}

constexpr FilterDesc gauss(float radius) {
  return { ResamplingFilterType::Gaussian, radius };
}

struct ResamplingTestEntry {
  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<int, 2> sizeWH,
                      FilterDesc filter,
                      double epsilon = 1)
    : ResamplingTestEntry(std::move(input)
    , std::move(reference), sizeWH, filter, filter, epsilon) {}

  ResamplingTestEntry(std::string input,
                      std::string reference,
                      std::array<int, 2> sizeWH,
                      FilterDesc fx,
                      FilterDesc fy,
                      double epsilon = 1)
    : input(std::move(input)), reference(std::move(reference)), epsilon(epsilon) {
    params[0].output_size = sizeWH[1];
    params[1].output_size = sizeWH[0];
    params[0].mag_filter = params[0].min_filter = fy;
    params[1].mag_filter = params[1].min_filter = fx;
  }

  std::string input, reference;
  ResamplingParams2D params;
  double epsilon = 1;
};

using ResamplingTestBatch = std::vector<ResamplingTestEntry>;

ResamplingTestBatch SingleImageBatch = {
  {
    "imgproc_test/containers.jpg", "imgproc_test/ref_out/containers_tri_300x300.png",
    { 300, 300 }, tri(), 5
  }
};

ResamplingTestBatch Batch1 = {
  {
    "imgproc_test/containers.jpg", "imgproc_test/ref_out/containers_tri_300x300.png",
    { 300, 300 }, tri(), 5
  },
  {
    "imgproc_test/score.png", "imgproc_test/ref_out/score_lanczos3.png",
    { 540, 250 }, lanczos(), 1
  },
  {
    "imgproc_test/containers.jpg", "imgproc_test/ref_out/containers_blurred.png",
    { 377, 480 }, gauss(12), 2
  }
};

std::ostream &operator<<(std::ostream &os, const FilterDesc fd) {
  const char *names[] = { "NN", "Linear", "Triangular", "Gaussian", "Lanczos3" };
  os << names[static_cast<int>(fd.type)];
  if (static_cast<int>(fd.type) > static_cast<int>(ResamplingFilterType::Linear) && fd.radius)
    os << "(r = " << fd.radius << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const ResamplingParams2D &params) {
  os  << "  Horizontal " << params[1].output_size << " px; "
      << " mag = " << params[1].mag_filter << " min = " << params[1].min_filter << "\n"
      << "  Vertical   " << params[0].output_size << " px; "
      << " mag = " << params[0].mag_filter << " min = " << params[0].min_filter << "\n";
  return os;
}

void PrintTo(const ResamplingTestEntry &entry, std::ostream *os) {
  *os << "Input: " << entry.input << "   ref:" << entry.reference << "\n  params:\n"
      << entry.params << "  Eps = " << entry.epsilon;
}

void PrintTo(const ResamplingTestBatch &batch, std::ostream *os) {
  *os << "{\n";
  bool first = true;
  for (auto &entry : batch) {
    if (first) { first = false;
    } else { *os << ",\n"; }
    PrintTo(entry, os);
  }
  *os << "\n}\n";
}

class BatchResamplingTest : public ::testing::Test,
                            public ::testing::WithParamInterface<ResamplingTestBatch> {
};

TEST_P(BatchResamplingTest, ResamplingImpl) {
  const ResamplingTestBatch &batch = GetParam();

  int N = batch.size();
  std::vector<cv::Mat> cv_img(N);
  std::vector<cv::Mat> cv_ref(N);
  std::vector<ResamplingParams2D> params(N);

  for (int i = 0; i < N; i++) {
    cv_img[i] = testing::data::image(batch[i].input.c_str());
    cv_ref[i] = testing::data::image(batch[i].reference.c_str());
    params[i] = batch[i].params;
  }

  ScratchpadAllocator scratch_alloc;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  SeparableResamplingGPUImpl<uint8_t, uint8_t> resampling;
  TestTensorList<uint8_t, 3> input, output;

  FilterDesc tri;
  tri.type = ResamplingFilterType::Triangular;
  FilterDesc lanczos;
  lanczos.type = ResamplingFilterType::Lanczos3;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < N; i++) {
    shapes.push_back(tensor_shape<3>(cv_img[i]));
  }
  input.reshape(shapes);
  OutListGPU<uint8_t, 3> in_tv = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_tv[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }

  auto req = resampling.Setup(ctx, in_tv, params);
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  scratch_alloc.Reserve(req.scratch_sizes);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tv = output.gpu();
  cudaMemset(out_tv.data, 0, out_tv.num_elements()*sizeof(*out_tv.data));

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_tv.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);

    EXPECT_EQ(in_tv.offsets[i], resampling.setup.sample_descs[i].offsets[0]);
    EXPECT_EQ(resampling.intermediate.offsets[i], resampling.setup.sample_descs[i].offsets[1]);
    EXPECT_EQ(out_tv.offsets[i], resampling.setup.sample_descs[i].offsets[2]);

    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[0], 1);
    EXPECT_GE(resampling.setup.sample_descs[i].block_count.pass[1], 1);
  }

  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  resampling.Run(ctx, out_tv, in_tv, params);
  for (int i = 0; i < N; i++) {
    auto ref_tensor = view_as_tensor<uint8_t, 3>(cv_ref[i]);
    auto out_tensor = output.cpu()[i];
    ASSERT_NO_FATAL_FAILURE(Check(out_tensor, ref_tensor, EqualEps(batch[i].epsilon)))
    << [&]() {
      cv::Mat tmp(out_tensor.shape[0], out_tensor.shape[1], CV_8UC3, out_tensor.data);
      std::string inp_name = batch[i].input;
      int ext = inp_name.rfind('.');
#ifdef WINVER
      constexpr char sep = '\\';
#else
      constexpr char sep = '/';
#endif
      int start = inp_name.rfind(sep) + 1;
      std::string dif_name = inp_name.substr(start, ext - start) + "_dif"+inp_name.substr(ext);
      cv::imwrite(dif_name, 127 + tmp - cv_ref[i]);
      return "Diff written to " + dif_name;
    }();
  }
}

TEST_P(BatchResamplingTest, ResamplingKernelAPI) {
  const ResamplingTestBatch &batch = GetParam();

  int N = batch.size();
  std::vector<cv::Mat> cv_img(N);
  std::vector<cv::Mat> cv_ref(N);
  std::vector<ResamplingParams2D> params(N);

  for (int i = 0; i < N; i++) {
    cv_img[i] = testing::data::image(batch[i].input.c_str());
    cv_ref[i] = testing::data::image(batch[i].reference.c_str());
    params[i] = batch[i].params;
  }

  ScratchpadAllocator scratch_alloc;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  using Kernel = ResampleGPU<uint8_t, uint8_t>;
  TestTensorList<uint8_t, 3> input, output;

  FilterDesc tri;
  tri.type = ResamplingFilterType::Triangular;
  FilterDesc lanczos;
  lanczos.type = ResamplingFilterType::Lanczos3;

  std::vector<TensorShape<3>> shapes;
  for (int i = 0; i < N; i++) {
    shapes.push_back(tensor_shape<3>(cv_img[i]));
  }
  input.reshape(shapes);
  OutListGPU<uint8_t, 3> in_tv = input.gpu();
  for (int i = 0; i < N; i++) {
    copy(in_tv[i], view_as_tensor<uint8_t, 3>(cv_img[i]));
  }

  auto req = Kernel::GetRequirements(ctx, in_tv, params);
  ASSERT_EQ(req.output_shapes.size(), 1);
  ASSERT_EQ(req.output_shapes[0].num_samples(), N);

  scratch_alloc.Reserve(req.scratch_sizes);

  output.reshape(req.output_shapes[0].to_static<3>());
  OutListGPU<uint8_t, 3> out_tv = output.gpu();
  cudaMemset(out_tv.data, 0, out_tv.num_elements()*sizeof(*out_tv.data));

  for (int i = 0; i < N; i++) {
    TensorShape<3> expected_shape = {
      params[i][0].output_size,
      params[i][1].output_size,
      in_tv.shape.tensor_shape_span(i)[2]
    };
    ASSERT_EQ(req.output_shapes[0].tensor_shape(i), expected_shape);
  }

  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  Kernel::Run(ctx, out_tv, in_tv, params);
  for (int i = 0; i < N; i++) {
    auto ref_tensor = view_as_tensor<uint8_t, 3>(cv_ref[i]);
    auto out_tensor = output.cpu()[i];
    ASSERT_NO_FATAL_FAILURE(Check(out_tensor, ref_tensor, EqualEps(batch[i].epsilon)))
    << [&]() {
      cv::Mat tmp(out_tensor.shape[0], out_tensor.shape[1], CV_8UC3, out_tensor.data);
      std::string inp_name = batch[i].input;
      int ext = inp_name.rfind('.');
#ifdef WINVER
      constexpr char sep = '\\';
#else
      constexpr char sep = '/';
#endif
      int start = inp_name.rfind(sep) + 1;
      std::string dif_name = inp_name.substr(start, ext - start) + "_dif"+inp_name.substr(ext);
      cv::imwrite(dif_name, 127 + tmp - cv_ref[i]);
      return "Diff written to " + dif_name;
    }();
  }
}


INSTANTIATE_TEST_CASE_P(SingleImage, BatchResamplingTest, ::testing::Values(SingleImageBatch));
INSTANTIATE_TEST_CASE_P(MultipleImages, BatchResamplingTest, ::testing::Values(Batch1));

}  // namespace kernels
}  // namespace dali
