// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_IMPL_CUH_
#define DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include "dali/core/tensor_view.h"
#include "dali/core/fast_div.h"
#include "dali/kernels/transpose/transpose_gpu_def.h"
#include "dali/core/static_switch.h"

namespace dali {
namespace kernels {
namespace transpose_impl {


struct TransposeInfo {
  void                *out;
  const void          *in;
  TransposeMethod     method;
  TensorShape<>       shape;
  SmallVector<int, 6> perm;
};

template <typename T>
struct TiledTransposeDesc {
  T *__restrict__ out;
  const T *__restrict__ in;
  uint64_t in_strides[kMaxNDim];
  // the output strides are permuted and it's this permutation that defines the whole operation
  uint64_t out_strides[kMaxNDim];
  fast_div<uint64_t> shape[kMaxNDim];

  uint32_t tiles_y;
  uint32_t tiles_per_block;
  uint32_t total_tiles;
  int ndim;
  int lanes;

  fast_div<uint32_t> tiles_x;
  fast_div<uint32_t> tiles_per_slice;
};

namespace transpose_shared {
  extern __shared__ uint8_t shared_tmp[];
}  // namespace transpose_shared


template <int ndim, typename T>
__device__ void TransposeTiledStatic(const TiledTransposeDesc<T> &desc) {
  T (*tmp)[kTileSize][kTileSize+1] =
      reinterpret_cast<T (*)[kTileSize][kTileSize+1]>(transpose_shared::shared_tmp);

  int lanes = desc.lanes;

  unsigned start_tile = blockIdx.x * desc.tiles_per_block;
  unsigned end_tile = min(desc.total_tiles, start_tile + desc.tiles_per_block);
  unsigned tile_in_slice;
  uint64_t fused_slice = ndim > 2
    ? div_mod(tile_in_slice, start_tile, desc.tiles_per_slice)
    : 0;

  uint64_t pos[kMaxNDim];

  for (int d = ndim - 3; d >= 0; d--) {
    fused_slice = div_mod(pos[d], fused_slice, desc.shape[d]);
  }

  unsigned tile_x, tile_y;
  tile_y = div_mod(tile_x, tile_in_slice, desc.tiles_x);
  pos[ndim - 1] = tile_x * kTileSize;
  pos[ndim - 2] = tile_y * kTileSize;

  T *out = desc.out;
  const T *in = desc.in;

  for (uint64_t tile = start_tile; tile < end_tile; tile++) {
    uint64_t in_ofs = 0, out_ofs = 0;
    for (int d = 0; d < ndim - 2; d++) {
      in_ofs  += desc.in_strides[d] * pos[d];
      out_ofs += desc.out_strides[d] * pos[d];
    }

    int64_t in_x  = pos[ndim-1] + threadIdx.x;
    int64_t in_y  = pos[ndim-2] + threadIdx.y;
    int64_t out_x = pos[ndim-1] + threadIdx.y;
    int64_t out_y = pos[ndim-2] + threadIdx.x;

    in_ofs  += desc.in_strides[ndim-2]  * in_y  + desc.in_strides[ndim-1]  * in_x;
    out_ofs += desc.out_strides[ndim-2] * out_y + desc.out_strides[ndim-1] * out_x;


    __syncthreads();
    int tile_w = min(static_cast<uint64_t>(kTileSize), desc.shape[ndim-1] - pos[ndim-1]);
    int tile_h = min(static_cast<uint64_t>(kTileSize), desc.shape[ndim-2] - pos[ndim-2]);
    if (threadIdx.x < tile_w) {
      for (int ty = threadIdx.y, dy = 0; ty < tile_h; ty += blockDim.y, dy += blockDim.y) {
        #pragma unroll(4)
        for (int lane = 0; lane < lanes; lane++) {
          tmp[lane][ty][threadIdx.x] = in[in_ofs + desc.in_strides[ndim-2]*dy + lane];
        }
      }
    }
    __syncthreads();

    if (threadIdx.x < tile_h) {
      for (int ty = threadIdx.y, dy = 0; ty < tile_w; ty += blockDim.y, dy += blockDim.y) {
        #pragma unroll(4)
        for (int lane = 0; lane < lanes; lane++)
          out[out_ofs + desc.out_strides[ndim-1]*dy + lane] = tmp[lane][threadIdx.x][ty];
      }
    }

    for (int d = ndim - 1; d >= 0; d--) {
      uint64_t delta = d < ndim-2 ? 1 : kTileSize;  // inner two dimensions are tiled
      pos[d] += delta;
      if (pos[d] < desc.shape[d])
        break;
      pos[d] = 0;
    }
  }
}

template <typename T>
__device__ void TransposeTiled(const TiledTransposeDesc<T> &desc) {
  VALUE_SWITCH(desc.ndim, static_ndim, (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    TransposeTiledStatic<static_ndim>(desc),
    {});
}

template <typename T>
__global__ void TransposeTiledSingle(TiledTransposeDesc<T> desc) {
  TransposeTiled(desc);
}

template <typename T>
__global__ void TransposeTiledBatch(const TiledTransposeDesc<T> *descs) {
  TransposeTiled(descs[blockIdx.y]);
}

template <typename T>
struct DeinterleaveDesc {
  T *__restrict__ out;
  const T *__restrict__ in;

  fast_div<uint64_t> in_strides[kMaxNDim];
  uint64_t out_strides[kMaxNDim];
  uint64_t size;
  int ndim;
};

template <typename T>
__device__ void TransposeDeinterleave(const DeinterleaveDesc<T> &desc) {
  const int tid = threadIdx.x;

  int ndim = desc.ndim;
  int lanes = desc.in_strides[ndim-2];

  uint64_t lane_stride = desc.out_strides[ndim-1];

  const uint64_t block_size = blockDim.x;
  uint64_t start_ofs = (blockIdx.x * block_size + tid) * lanes;
  uint64_t grid_stride = gridDim.x * block_size * lanes;

  T *out = desc.out;
  const T *in = desc.in;

  for (uint64_t in_ofs = start_ofs; in_ofs < desc.size; in_ofs += grid_stride) {
    uint64_t out_ofs = 0;
    uint64_t tmp_idx = in_ofs;
    for (int d = 0; d < ndim - 1; d++) {
      uint64_t a = div_mod(tmp_idx, tmp_idx, desc.in_strides[d]);
      out_ofs += a * desc.out_strides[d];
    }

    for (int lane = 0; lane < lanes; lane++) {
      out[out_ofs + lane * lane_stride] = __ldg(&in[in_ofs + lane]);
    }
  }
}

template <typename T>
struct GenericTransposeDesc {
  T *__restrict__ out;
  const T *__restrict__ in;

  // these are input strides for Deinterleave and output strides for otherwise
  fast_div<uint64_t> out_strides[kMaxNDim];
  // ...and vice versa
  uint64_t in_strides[kMaxNDim];
  uint64_t size;
  int ndim;
};

template <typename T>
__device__ void TransposeGeneric(const GenericTransposeDesc<T> &desc) {
  const int tid = threadIdx.x;

  int ndim = desc.ndim;

  const uint64_t block_size = blockDim.x;
  uint64_t start_ofs = blockIdx.x * block_size + tid;
  uint64_t grid_stride = gridDim.x * block_size;

  T *out = desc.out;
  const T *in = desc.in;

  for (uint64_t out_ofs = start_ofs; out_ofs < desc.size; out_ofs += grid_stride) {
    uint64_t in_ofs = 0;
    uint64_t tmp_idx = out_ofs;
    for (int d = 0; d < ndim - 1; d++) {
      uint64_t a = div_mod(tmp_idx, tmp_idx, desc.out_strides[d]);
      in_ofs += a * desc.in_strides[d];
    }
    in_ofs += tmp_idx * desc.in_strides[ndim-1];

    out[out_ofs] = in[in_ofs];
  }
}

template <typename T>
__global__ void TransposeDeinterleaveSingle(DeinterleaveDesc<T> desc) {
  TransposeDeinterleave(desc);
}

template <typename T>
__global__ void TransposeDeinterleaveBatch(const DeinterleaveDesc<T> *descs) {
  TransposeDeinterleave(descs[blockIdx.y]);
}

template <typename T>
__global__ void TransposeGenericSingle(GenericTransposeDesc<T> desc) {
  TransposeGeneric(desc);
}

template <typename T>
__global__ void TransposeGenericBatch(const GenericTransposeDesc<T> *descs) {
  TransposeGeneric(descs[blockIdx.y]);
}


}  // namespace transpose_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TRANSPOSE_TRANSPOSE_GPU_IMPL_CUH_
