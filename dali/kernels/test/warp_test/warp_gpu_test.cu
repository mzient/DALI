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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <vector>
#include "dali/kernels/imgproc/warp_gpu.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/kernels/test/kernel_test_utils.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/test/mat2tensor.h"
#include "dali/kernels/test/test_tensors.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/alloc.h"
#include "dali/test/dali_test_config.h"
#include "dali/core/geom/transform.h"

namespace dali {
namespace kernels {

void IsWarpKernelValid() {
  check_kernel<WarpGPU<AffineMapping2D, 2, float, uint8_t, float, DALI_INTERP_LINEAR>>();
}

namespace {

template <int dim>
struct BlockMap {
  unsigned end;
  std::map<unsigned, BlockMap<dim-1>> inner;
};

template <>
struct BlockMap<0> {
  unsigned end;
  // dummy - to avoid specialization
  const bool inner = false;
};

template <int dim>
bool operator==(const BlockMap<dim> &a, const BlockMap<dim> &b) {
  return a.end == b.end && a.inner == b.inner;
}

inline void ValidateBlockMap(const BlockMap<0> &map, const TensorShape<0> &shape) {}

/// @brief Check that the output shape is covered with rectangular grid.
///
/// The grid cells must be aligned between rows/slices, but don't have to be uniform
/// - typically the last cell will be smaller and that's expected.
template <int dim>
void ValidateBlockMap(const BlockMap<dim> &map, const TensorShape<dim> &shape) {
  ASSERT_FALSE(map.inner.empty());
  unsigned i = 0;
  for (auto &p : map.inner) {
    ASSERT_EQ(p.first, i) << "Blocks don't cover the image";
    ASSERT_GT(p.second.end, i) << "Block end coordinate must be greater than start";
    i = p.second.end;
  }
  EXPECT_EQ(i, shape[0])
    << (i < shape[0] ? "Block does not cover whole image" : "Block exceeds image size");

  const BlockMap<dim-1> *first_slice = 0;
  for (auto &p : map.inner) {
    if (first_slice) {
      EXPECT_EQ(p.second.inner, first_slice->inner) << "Inner block layout must be uniform";
    } else {
      first_slice = &p.second;
      // Validate the first slice recurisvely - the remaining slices should be equal and
      // therefore don't require validation.
      ValidateBlockMap(p.second, shape.template last<dim-1>());
    }
  }
}

}  // namespace

TEST(WarpSetup, Setup_Blocks) {
  TensorListShape<3> TLS({
    { 480, 640, 3 },
    { 768, 1024, 3 },
    { 600, 800, 3 },
    { 720, 1280, 3 },
    { 480, 864, 3 },
    { 576, 720, 3 }
  });

  warp::WarpSetup<2> setup;
  setup.Setup(TLS);
  int prev = -1;
  BlockMap<2> map;
  for (auto &blk : setup.Blocks()) {
    if (blk.sample_idx != prev) {
      if (prev != -1) {
        ValidateBlockMap(map, TLS[prev].first<2>());
      }
      prev = blk.sample_idx;
      map = {};
    }
    auto &b = map.inner[blk.start.y];
    b.end = blk.end.y;
    b.inner[blk.start.x].end = blk.end.x;
  }
  if (prev != -1)
    ValidateBlockMap(map, TLS[prev].first<2>());
}

TEST(WarpGPU, Affine_Transpose_ForceVariable) {
  AffineMapping2D mapping_cpu = mat2x3{{
    { 0, 1, 0 },
    { 1, 0, 0 }
  }};

  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/alley.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<AllocType::GPU>(cpu_img);
  auto img_tensor = gpu_img.first;

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(1, 3);
  in_list.shape.set_tensor_shape(0, img_tensor.shape);
  in_list.data[0] = img_tensor.data;

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, BorderClamp, DALI_INTERP_NN> warp;

  ScratchpadAllocator scratch_alloc;

  auto mapping_gpu = memory::alloc_unique<AffineMapping2D>(AllocType::GPU, 1);
  TensorShape<2> out_shape = { img_tensor.shape[1], img_tensor.shape[0] };
  KernelContext ctx = {};
  auto out_shapes_hw = make_span<1>(&out_shape);
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { 1 });
  copy(mappings, make_tensor_cpu<1>(&mapping_cpu, { 1 }));

  auto out_shapes = warp.GetOutputShape(in_list.shape, out_shapes_hw);
  KernelRequirements req = warp.WarpSetup::Setup(out_shapes, true);
  scratch_alloc.Reserve(req.scratch_sizes);
  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  warp.Run(ctx, out.gpu(0), in_list, mappings, out_shapes_hw);

  auto cpu_out = out.cpu(0)[0];
  cudaDeviceSynchronize();
  ASSERT_EQ(cpu_out.shape[0], img_tensor.shape[1]);
  ASSERT_EQ(cpu_out.shape[1], img_tensor.shape[0]);
  ASSERT_EQ(cpu_out.shape[2], 3);

  for (int y = 0; y < cpu_out.shape[0]; y++) {
    for (int x = 0; x < cpu_out.shape[1]; x++) {
      for (int c = 0; c < 3; c++) {
        EXPECT_EQ(*cpu_out(y, x, c), *cpu_img(x, y, c));
      }
    }
  }
}

TEST(WarpGPU, Affine_Transpose_Single) {
  AffineMapping2D mapping_cpu = mat2x3{{
    { 0, 1, 0 },
    { 1, 0, 0 }
  }};

  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/alley.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<AllocType::GPU>(cpu_img);
  auto img_tensor = gpu_img.first;

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(1, 3);
  in_list.shape.set_tensor_shape(0, img_tensor.shape);
  in_list.data[0] = img_tensor.data;

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, BorderClamp, DALI_INTERP_NN> warp;

  ScratchpadAllocator scratch_alloc;

  auto mapping_gpu = memory::alloc_unique<AffineMapping2D>(AllocType::GPU, 1);
  TensorShape<2> out_shape = { img_tensor.shape[1], img_tensor.shape[0] };
  KernelContext ctx = {};
  auto out_shapes_hw = make_span<1>(&out_shape);
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { 1 });
  copy(mappings, make_tensor_cpu<1>(&mapping_cpu, { 1 }));

  KernelRequirements req = warp.Setup(ctx, in_list, mappings, out_shapes_hw);
  scratch_alloc.Reserve(req.scratch_sizes);
  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  warp.Run(ctx, out.gpu(0), in_list, mappings, out_shapes_hw);

  auto cpu_out = out.cpu(0)[0];
  cudaDeviceSynchronize();
  usleep(100000);
  ASSERT_EQ(cpu_out.shape[0], img_tensor.shape[1]);
  ASSERT_EQ(cpu_out.shape[1], img_tensor.shape[0]);
  ASSERT_EQ(cpu_out.shape[2], 3);

  int errors = 0;
  int printed = 0;
  for (int y = 0; y < cpu_out.shape[0]; y++) {
    for (int x = 0; x < cpu_out.shape[1]; x++) {
      for (int c = 0; c < 3; c++) {
        if (*cpu_out(y, x, c) != *cpu_img(x, y, c)) {
          if (errors++ < 100) {
            printed++;
            EXPECT_EQ(*cpu_out(y, x, c), *cpu_img(x, y, c))
              << "@ x = " << x << " y = " << y << " c = " << c;
          }
        }
      }
    }
  }
  if (printed != errors) {
    FAIL() << (errors - printed) << " more erors.";
  }
}

/// @brief Apply correction of pixel centers and convert the mapping to
///        OpenCV matrix type.
inline cv::Matx<float, 2, 3> AffineToCV(const AffineMapping2D &mapping) {
  vec2 translation = mapping({0.5f, 0.5f}) - vec2(0.5f, 0.5f);
  mat2x3 tmp = mapping.transform;
  tmp.set_col(2, translation);

  cv::Matx<float, 2, 3> cv_transform;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      cv_transform(i, j) = tmp(i, j);
  return cv_transform;
}

TEST(WarpGPU, Affine_RotateScale_Single) {
  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/dots.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<AllocType::GPU>(cpu_img);
  auto img_tensor = gpu_img.first;

  vec2 center(cv_img.cols * 0.5f, cv_img.rows * 0.5f);

  int scale = 10;
  auto tr = translation(center) * rotation2D(-M_PI/4) *
            translation(-center) * scaling(vec2(1.0f/scale, 1.0f/scale));
  AffineMapping2D mapping_cpu = sub<2, 3>(tr, 0, 0);

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(1, 3);
  in_list.shape.set_tensor_shape(0, img_tensor.shape);
  in_list.data[0] = img_tensor.data;

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, uint8_t, DALI_INTERP_LINEAR> warp;

  ScratchpadAllocator scratch_alloc;

  auto mapping_gpu = memory::alloc_unique<AffineMapping2D>(AllocType::GPU, 1);
  TensorShape<2> out_shape = { img_tensor.shape[0] * scale, img_tensor.shape[1] * scale };
  KernelContext ctx = {};
  auto out_shapes_hw = make_span<1>(&out_shape);
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { 1 });
  copy(mappings, make_tensor_cpu<1>(&mapping_cpu, { 1 }));

  auto out_shapes = warp.GetOutputShape(in_list.shape, out_shapes_hw);
  KernelRequirements req = warp.WarpSetup::Setup(out_shapes, true);
  scratch_alloc.Reserve(req.scratch_sizes);
  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  warp.Run(ctx, out.gpu(0), in_list, mappings, out_shapes_hw, 255);

  auto cpu_out = out.cpu(0)[0];
  cudaDeviceSynchronize();
  ASSERT_EQ(cpu_out.shape[0], out_shapes_hw[0][0]);
  ASSERT_EQ(cpu_out.shape[1], out_shapes_hw[0][1]);
  ASSERT_EQ(cpu_out.shape[2], 3);

  cv::Mat cv_out(cpu_out.shape[0], cpu_out.shape[1], CV_8UC3, cpu_out.data);

  cv::Matx<float, 2, 3> cv_transform = AffineToCV(mapping_cpu);

  cv::Mat cv_ref;
  cv::warpAffine(cv_img, cv_ref,
                 cv_transform, cv::Size(out_shape[1], out_shape[0]),
                 cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255, 255));
  auto ref_img = view_as_tensor<uint8_t>(cv_ref);
  Check(cpu_out, ref_img, EqualEps(8));
  if (HasFailure()) {
    cv::imwrite("Warp_Affine_RotateScale_out.png", cv_out);
    cv::imwrite("Warp_Affine_RotateScale_ref.png", cv_ref);
    cv::Mat diff;
    cv::absdiff(cv_out, cv_ref, diff);
    cv::imwrite("Warp_Affine_RotateScale_diff.png", diff);
  }
}


TEST(WarpGPU, Affine_RotateScale_Uniform) {
  cv::Mat cv_img = cv::imread(testing::dali_extra_path() + "/db/imgproc/dots.png");
  auto cpu_img = view_as_tensor<uint8_t>(cv_img);
  auto gpu_img = copy<AllocType::GPU>(cpu_img);
  auto img_tensor = gpu_img.first;

  vec2 center(cv_img.cols * 0.5f, cv_img.rows * 0.5f);

  const int samples = 10;
  std::vector<AffineMapping2D> mapping_cpu(samples);
  int scale = 10;

  TensorListView<StorageGPU, uint8_t, 3> in_list;
  in_list.resize(samples, 3);
  for (int i = 0; i < samples; i++) {
    in_list.shape.set_tensor_shape(i, img_tensor.shape);
    in_list.data[i] = img_tensor.data;

    auto tr = translation(center) * rotation2D(-2*M_PI * i / samples) *
              translation(-center) * scaling(vec2(1.0f/scale, 1.0f/scale));
    mapping_cpu[i] = sub<2, 3>(tr, 0, 0);
  }

  WarpGPU<AffineMapping2D, 2, uint8_t, uint8_t, uint8_t, DALI_INTERP_LINEAR> warp;

  ScratchpadAllocator scratch_alloc;

  auto mapping_gpu = memory::alloc_unique<AffineMapping2D>(AllocType::GPU, samples);
  TensorShape<2> out_shape = { img_tensor.shape[0] * scale, img_tensor.shape[1] * scale };
  KernelContext ctx = {};
  std::vector<TensorShape<2>> out_shapes_hw(samples);
  for (int i = 0; i < samples; i++)
    out_shapes_hw[i] = out_shape;
  auto mappings = make_tensor_gpu<1>(mapping_gpu.get(), { samples });
  copy(mappings, make_tensor_cpu<1>(mapping_cpu.data(), { samples }));

  auto out_shapes = warp.GetOutputShape(in_list.shape, make_span(out_shapes_hw));
  KernelRequirements req = warp.WarpSetup::Setup(out_shapes, true);
  scratch_alloc.Reserve(req.scratch_sizes);
  TestTensorList<uint8_t, 3> out;
  out.reshape(req.output_shapes[0].to_static<3>());
  auto scratchpad = scratch_alloc.GetScratchpad();
  ctx.scratchpad = &scratchpad;
  warp.Run(ctx, out.gpu(0), in_list, mappings, make_span(out_shapes_hw), 255);
  cudaDeviceSynchronize();

  for (int i = 0; i < samples; i++) {
    auto cpu_out = out.cpu(0)[i];
    ASSERT_EQ(cpu_out.shape[0], out_shapes_hw[i][0]);
    ASSERT_EQ(cpu_out.shape[1], out_shapes_hw[i][1]);
    ASSERT_EQ(cpu_out.shape[2], 3);

    cv::Mat cv_out(cpu_out.shape[0], cpu_out.shape[1], CV_8UC3, cpu_out.data);

    cv::Matx<float, 2, 3> cv_transform = AffineToCV(mapping_cpu[i]);

    cv::Mat cv_ref;
    cv::warpAffine(cv_img, cv_ref,
                  cv_transform, cv::Size(out_shape[1], out_shape[0]),
                  cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,
                  cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255, 255));
    auto ref_img = view_as_tensor<uint8_t>(cv_ref);
    Check(cpu_out, ref_img, EqualEps(8));
    char out_name[64];
    char ref_name[64];
    char diff_name[64];
    snprintf(out_name, sizeof(out_name), "Warp_Affine_RotateScale_%i_out.png", i);
    snprintf(ref_name, sizeof(ref_name), "Warp_Affine_RotateScale_%i_ref.png", i);
    snprintf(diff_name, sizeof(diff_name), "Warp_Affine_RotateScale_%i_diff.png", i);
    if (HasFailure()) {
      cv::imwrite(out_name, cv_out);
      cv::imwrite(ref_name, cv_ref);
      cv::Mat diff;
      cv::absdiff(cv_out, cv_ref, diff);
      cv::imwrite(diff_name, diff);
    }
  }
}

}  // namespace kernels
}  // namespace dali
