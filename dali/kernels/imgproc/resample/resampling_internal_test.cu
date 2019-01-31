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
#include "dali/kernels/tensor_view.h"
#include "dali/kernels/tensor_shape_print.h"
#include "dali/kernels/imgproc/resample/resampling_impl.cuh"

namespace dali {
namespace kernels {

template <typename Dst, typename Src>
__global__ void ResampleHorzTestKernel(
    Dst *out, int out_stride, int out_w,
    const Src *in, int in_stride, int in_w, int in_h, int channels,
    ResamplingFilter filter, int radius) {
  float scale = (float)in_w / out_w;

  int x0 = blockIdx.x * out_w / gridDim.x;
  int x1 = (blockIdx.x + 1) * out_w / gridDim.x;
  int y0 = blockIdx.y * in_h / gridDim.y;
  int y1 = (blockIdx.y + 1) * in_h / gridDim.y;
  ResampleHorz(x0, x1, y0, y1, 0, scale,
               out, out_stride, in, in_stride, in_w,
               channels, filter, 2*radius+1, -radius);
}

template <typename Dst, typename Src>
__global__ void ResampleVertTestKernel(
    Dst *out, int out_stride, int out_h,
    const Src *in, int in_stride, int in_w, int in_h, int channels,
    ResamplingFilter filter, int radius) {
  float scale = (float)in_h / out_h;

  int x0 = blockIdx.x * in_w / gridDim.x;
  int x1 = (blockIdx.x + 1) * in_w / gridDim.x;
  int y0 = blockIdx.y * out_h / gridDim.y;
  int y1 = (blockIdx.y + 1) * out_h / gridDim.y;
  ResampleVert(x0, x1, y0, y1, 0, scale,
               out, out_stride, in, in_stride, in_h,
               channels, filter, 2*radius+1, -radius);
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


TEST(ResampleHorz, TestGaussian) {
  //auto cv_img = cv::imread("../DALI_data/in/0/rgb_0.jpg");
  auto cv_img = cv::imread("test.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outW = W/2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, H * outW * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { H, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  ResamplingFilter filter = (*filters)[1];

  int radius = 40;
  filter.scale = (float)filter.size/(2*radius+1);

  for (int i=0; i<100; i++) {
    ResampleHorzTestKernel<<<1, dim3(32, 24)>>>(
      img_out.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      filter, radius);
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(H, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  cv::imwrite("resample_horz.png", out);
}

TEST(ResampleVert, TestGaussian) {
  //auto cv_img = cv::imread("../DALI_data/in/0/rgb_0.jpg");
  auto cv_img = cv::imread("test.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outH = H/2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, outH * W * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { outH, W, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  ResamplingFilter filter = (*filters)[1];

  int radius = 40;
  filter.scale = (float)filter.size/(2*radius+1);

  for (int i=0; i<100; i++) {
    ResampleVertTestKernel<<<1, dim3(32, 24)>>>(
      img_out.data, W*channels, outH, img_in.data, W*channels, W, H, channels,
      filter, radius);
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(outH, W, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  cv::imwrite("resample_vert.png", out);
}

TEST(ResampleHV, TestGaussian) {
  //auto cv_img = cv::imread("../DALI_data/in/0/rgb_0.jpg");
  auto cv_img = cv::imread("wall.jpg");
  //auto cv_img = cv::imread("test.png");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  double scaleX = 3.17;
  double scaleY = 2.57;
  int outH = H/scaleY;
  int outW = W/scaleX;
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
  ResamplingFilter filter = (*filters)[1];

  int radiusX = ceil(scaleX);
  int radiusY = ceil(scaleY);

  for (int i=0; i<100; i++) {
    filter.scale = (float)filter.size/(2*scaleX+1);
    ResampleHorzTestKernel<<<1, dim3(32, 24)>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      filter, radiusX);
    filter.scale = (float)filter.size/(2*scaleY+1);
    ResampleVertTestKernel<<<1, dim3(32, 24)>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      filter, radiusY);
    cudaDeviceSynchronize();
  }

  cv::Mat out;
  out.create(outH, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  cv::imwrite("resample_out.png", out);
}

inline constexpr int divUp(int total, int grain) {
  return (total + grain - 1) / grain;
}

TEST(ResampleHV, Blur) {
  //auto cv_img = cv::imread("../DALI_data/in/0/rgb_0.jpg");
  auto cv_img = cv::imread("wall.jpg");
  //auto cv_img = cv::imread("test.png");
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
  ResamplingFilter filter = (*filters)[1];

  int radiusX = 3;
  int radiusY = 3;

  for (int i=0; i<100; i++) {
    filter.scale = (float)filter.size/(2*radiusX+1);
    ResampleHorzTestKernel<<<dim3(divUp(outW, 32), 1), dim3(32, 24)>>>(
      img_tmp.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
      filter, radiusX);
    filter.scale = (float)filter.size/(2*radiusY+1);
    ResampleVertTestKernel<<<dim3(1, divUp(outH, 24)), dim3(32, 24)>>>(
      img_out.data, outW*channels, outH, img_tmp.data, outW*channels, outW, H, channels,
      filter, radiusY);
  }

  cv::Mat out;
  out.create(outH, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  cv::imwrite("blur_out.png", out);
}

}  // namespace dali
}  // namespace kernels
