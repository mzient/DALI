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
#include "dali/kernels/alloc.h"
#include "dali/kernels/test/mat2tensor.h"
#include "dali/kernels/test/test_data.h"
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/imgproc/resample/resampling_impl.cuh"
#include "dali/kernels/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

template <typename Dst, typename Src>
__global__ void ResampleHorzTestKernel(
    Dst *out, int out_stride, int out_w,
    const Src *in, int in_stride, int in_w, int in_h, int channels,
    ResamplingFilter filter, int support) {
  float scale = (float)in_w / out_w;

  int x0 = blockIdx.x * out_w / gridDim.x;
  int x1 = (blockIdx.x + 1) * out_w / gridDim.x;
  int y0 = blockIdx.y * in_h / gridDim.y;
  int y1 = (blockIdx.y + 1) * in_h / gridDim.y;
  ResampleHorz(
    x0, x1, y0, y1, 0, scale,
    out, out_stride, in, in_stride, in_w,
    channels, filter, support);
}

template <typename Dst, typename Src>
__global__ void ResampleVertTestKernel(
    Dst *out, int out_stride, int out_h,
    const Src *in, int in_stride, int in_w, int in_h, int channels,
    ResamplingFilter filter, int support) {
  float scale = (float)in_h / out_h;

  int x0 = blockIdx.x * in_w / gridDim.x;
  int x1 = (blockIdx.x + 1) * in_w / gridDim.x;
  int y0 = blockIdx.y * out_h / gridDim.y;
  int y1 = (blockIdx.y + 1) * out_h / gridDim.y;
  ResampleVert(
    x0, x1, y0, y1, 0, scale,
    out, out_stride, in, in_stride, in_h,
    channels, filter, support);
}

template <typename T, typename U, int ndim1, int ndim2>
void copy(const TensorView<StorageGPU, T, ndim1> &out, const TensorView<StorageCPU, U, ndim2> &in, cudaStream_t stream = 0) {
  static_assert(sizeof(T) == sizeof(U), "Tensor elements must be of equal size");
  assert(in.shape == out.shape);
  cudaMemcpyAsync(out.data, in.data, in.num_elements() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T, typename U, int ndim1, int ndim2>
void copy(const TensorView<StorageCPU, T, ndim1> &out, const TensorView<StorageGPU, U, ndim2> &in, cudaStream_t stream = 0) {
  static_assert(sizeof(T) == sizeof(U), "Tensor elements must be of equal size");
  assert(in.shape == out.shape);
  cudaMemcpyAsync(out.data, in.data, in.num_elements() * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T, typename U, int ndim1, int ndim2>
void copy(const TensorView<StorageGPU, T, ndim1> &out, const TensorView<StorageGPU, U, ndim2> &in, cudaStream_t stream = 0) {
  static_assert(sizeof(T) == sizeof(U), "Tensor elements must be of equal size");
  assert(in.shape == out.shape);
  cudaMemcpyAsync(out.data, in.data, in.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T, typename U, int ndim1, int ndim2>
void copy(const TensorView<StorageCPU, T, ndim1> &out, const TensorView<StorageCPU, U, ndim2> &in, cudaStream_t stream = 0) {
  static_assert(sizeof(T) == sizeof(U), "Tensor elements must be of equal size");
  assert(in.shape == out.shape);
  cudaMemcpy(out.data, in.data, in.num_elements() * sizeof(T), cudaMemcpyHostToHost);
}


TEST(Resample, HorizontalGaussian) {
  auto cv_img = testing::data::image("imgproc_test/checkerboard.png");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/resample_horz.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outW = W / 2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, H * outW * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { H, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  ResamplingFilter filter = (*filters)[1];

  int radius = 40;
  filter.rescale(2*radius+1);

  for (int i=0; i<100; i++) {
    ResampleHorzTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      filter, filter.support());
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(H, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(1))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("resample_horz_dif.png", diff);
    return "Test failed. Absolute difference image saved to resample_horz_dif.png";
  }();
}

TEST(Resample, VerticalGaussian) {
  auto cv_img = testing::data::image("imgproc_test/checkerboard.png");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/resample_vert.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H / 2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outH * W * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { outH, W, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  ResamplingFilter filter = (*filters)[1];

  int radius = 40;
  filter.rescale(2*radius+1);

  for (int i=0; i<100; i++) {
    ResampleVertTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, W*channels, outH, img_in.data, W*channels, W, H, channels,
      filter, filter.support());
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(outH, W, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(1))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("resample_vert_dif.png", diff);
    return "Test failed. Absolute difference image saved to resample_hv_dif.png";
  }();
}

TEST(Resample, SeparableGauss) {
  auto cv_img = testing::data::image("imgproc_test/moire2.png");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/resample_out.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H-3;
  int outW = W-1;
  double scaleX = 1.0 * W / outW;
  double scaleY = 1.0 * H / outH;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_tmp = memory::alloc_unique<float>(AllocType::GPU, outW * H * channels);
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outW * outH * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  TensorView<StorageGPU, float, 3> img_tmp;
  img_in = { gpu_mem_in.get(), img.shape };
  img_tmp = { gpu_mem_tmp.get(), { H, outW, channels } };
  img_out = { gpu_mem_out.get(), { outH, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  auto fx = filters->Gaussian(scaleX - 0.3f);
  auto fy = filters->Gaussian(scaleY - 0.3f);

  for (int i=0; i<100; i++) {
    ResampleHorzTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      fx, fx.support());
    ResampleVertTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      fy, fy.support());
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(outH, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(1))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("resample_hv_dif.png", diff);
    return "Test failed. Absolute difference image saved to resample_hv_dif.png";
  }();
}


TEST(Resample, SeparableTriangular) {
  auto cv_img = testing::data::image("imgproc_test/containers.jpg");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/containers_tri_300x300.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = 300;//H/3;
  int outW = 300;//W/2;
  double scaleX = 1.0 * W / outW;
  double scaleY = 1.0 * H / outH;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_tmp = memory::alloc_unique<float>(AllocType::GPU, outW * H * channels);
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outW * outH * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  TensorView<StorageGPU, float, 3> img_tmp;
  img_in = { gpu_mem_in.get(), img.shape };
  img_tmp = { gpu_mem_tmp.get(), { H, outW, channels } };
  img_out = { gpu_mem_out.get(), { outH, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  auto fx = filters->Triangular(1);
  fx.rescale(scaleX);
  auto fy = filters->Triangular(1);
  fy.rescale(scaleX);

  for (int i=0; i<100; i++) {
    ResampleHorzTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      fx, fx.support());
    cudaDeviceSynchronize();
    ResampleVertTestKernel<<<1, dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      fy, fy.support());
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(outH, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(2))) <<
  cv::imwrite("containers_tri.png", out);
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("resample_tri_dif.png", diff);
    return "Test failed. Absolute difference image saved to resample_tri_dif.png";
  }();
}

inline constexpr int divUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

TEST(GaussianBlur, OneImage) {
  auto cv_img = testing::data::image("imgproc_test/containers.jpg");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/containers_blurred.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H;
  int outW = W;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_tmp = memory::alloc_unique<float>(AllocType::GPU, outW * H * channels);
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outW * outH * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  TensorView<StorageGPU, float, 3> img_tmp;
  img_in = { gpu_mem_in.get(), img.shape };
  img_tmp = { gpu_mem_tmp.get(), { H, outW, channels } };
  img_out = { gpu_mem_out.get(), { outH, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);

  float sigmaX = 6.0f;
  float sigmaY = 6.0f;

  ResamplingFilter fx = filters->Gaussian(sigmaX);
  ResamplingFilter fy = filters->Gaussian(sigmaY);

  for (int i=0; i<100; i++) {
    ResampleHorzTestKernel<<<dim3(divUp(outW, 32), 1), dim3(32, 24), ResampleSharedMemSize>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      fx, fx.support());
    ResampleVertTestKernel<<<dim3(1, divUp(outH, 24)), dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      fy, fy.support());
  }

  cv::Mat out;
  out.create(outH, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(2))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("blur_dif.png", diff);
    return "Test failed. Difference image saved to blur_dif.png";
  }();
}

TEST(GaussianBlur, DISABLED_ProgressiveOutputs) {
  auto cv_img = testing::data::image("imgproc_test/containers.jpg");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H;
  int outW = W;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_tmp = memory::alloc_unique<float>(AllocType::GPU, outW * H * channels);
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outW * outH * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  TensorView<StorageGPU, float, 3> img_tmp;
  img_in = { gpu_mem_in.get(), img.shape };
  img_tmp = { gpu_mem_tmp.get(), { H, outW, channels } };
  img_out = { gpu_mem_out.get(), { outH, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);

  for (int i=0; i<10; i++) {
    float sigmaX = powf(1.10f, i) * 0.5f;
    float sigmaY = powf(1.10f, i) * 0.5f;

    ResamplingFilter fx = filters->Gaussian(sigmaX);
    ResamplingFilter fy = filters->Gaussian(sigmaY);

    ResampleHorzTestKernel<<<dim3(divUp(outW, 32), 1), dim3(32, 24), ResampleSharedMemSize>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      fx, fx.support());
    ResampleVertTestKernel<<<dim3(1, divUp(outH, 24)), dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      fy, fy.support());

    cv::Mat out;
    out.create(outH, outW, CV_8UC3);
    auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
    copy(img_out_cpu, img_out);
    cudaDeviceSynchronize();
    char name[64];
    sprintf(name, "blur_%i.png", i);
    cv::imwrite(name, out);
  }

}


TEST(Lanczos, OneImage) {
  auto cv_img = testing::data::image("imgproc_test/score.png");
  auto cv_ref = testing::data::image("imgproc_test/ref_out/score_lanczos3.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H*5;
  int outW = W*5;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_tmp = memory::alloc_unique<float>(AllocType::GPU, outW * H * channels);
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outW * outH * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  TensorView<StorageGPU, float, 3> img_tmp;
  img_in = { gpu_mem_in.get(), img.shape };
  img_tmp = { gpu_mem_tmp.get(), { H, outW, channels } };
  img_out = { gpu_mem_out.get(), { outH, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);

  ResamplingFilter fx = filters->Lanczos3();
  ResamplingFilter fy = filters->Lanczos3();

  for (int i=0; i<100; i++) {
    ResampleHorzTestKernel<<<dim3(divUp(outW, 32), 1), dim3(32, 24), ResampleSharedMemSize>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      fx, fx.support());
    ResampleVertTestKernel<<<dim3(1, divUp(outH, 24)), dim3(32, 24), ResampleSharedMemSize>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      fy, fy.support());
  }

  cv::Mat out;
  out.create(outH, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  auto img_ref_cpu = view_as_tensor<uint8_t, 3>(cv_ref);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  EXPECT_NO_FATAL_FAILURE(Check(img_out_cpu, img_ref_cpu, EqualEps(1))) <<
  [&]() {
    cv::Mat diff;
    cv::absdiff(out, cv_ref, diff);
    cv::imwrite("lanczos3_dif.png", diff);
    return "Test failed. Difference image saved to blur_dif.png";
  }();
}

}  // namespace dali
}  // namespace kernels
