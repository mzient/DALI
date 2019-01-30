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
    const Src *in, int in_stride, int in_w, int h, int channels,
    ResamplingFilter filter, int radius) {
  ResampleHorz(out, out_stride, out_w, in, in_stride, in_w, h, channels, filter, radius);
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
  auto cv_img = cv::imread("../DALI_data/in/0/rgb_0.jpg");
  ASSERT_FALSE(cv_img.empty()) << "Couldn't load the image";
  TensorView<StorageCPU, const uint8_t, 3> img;
  ASSERT_NO_FATAL_FAILURE((img = view_as_tensor<uint8_t>(cv_img)));

  int channels = img.shape[2];
  ASSERT_EQ(channels, 3);
  int H = img.shape[0];
  int W = img.shape[1];
  int outW = img.shape[1] / 2;
  auto gpu_mem_in = memory::alloc_unique<uint8_t>(AllocType::GPU, img.num_elements());
  auto gpu_mem_out = memory::alloc_unique<uint8_t>(AllocType::GPU, H * outW * channels);

  TensorView<StorageGPU, uint8_t, 3> img_in, img_out;
  img_in = { gpu_mem_in.get(), img.shape };
  img_out = { gpu_mem_out.get(), { H, outW, channels } };

  copy(img_in, img);

  auto filters = GetResamplingFilters(0);
  ResamplingFilter filter = (*filters)[1];

  filter.scale = 1;

  ResampleHorzTestKernel<<<1, dim3(32, 16)>>>(
    img_out.data, outW*channels, outW, img_in.data, W*channels, W, H, channels,
    filter, 4);

  cv::Mat out;
  out.create(H, outW, CV_8UC3);
  auto img_out_cpu = view_as_tensor<uint8_t, 3>(out);
  copy(img_out_cpu, img_out);
  cudaDeviceSynchronize();
  cv::imwrite("resize_output.png", out);
}

}  // namespace dali
  }  // namespace kernels
