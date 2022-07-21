// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_
#define DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_

#include <typeinfo>
#include <vector>
#include "dali/core/format.h"
#include "dali/imgcodec/image_decoder.h"

namespace dali {
namespace imgcodec {

template <typename Actual>
class DLL_PUBLIC ImageDecoderImpl : public ImageDecoderInstance {
 public:
  ImageDecoderImpl(int device_id, ThreadPool *tp) : device_id_(device_id), tp_(tp) {
  }

  bool CanDecode(ImageSource *in, DecodeParams opts) override { return true; }

  std::vector<bool> CanDecode(cspan<ImageSource *> in, cspan<DecodeParams> opts) override {
    assert(opts.size() == in.size());
    std::vector<bool> ret(in.size());
    for (int i = 0; i < in.size(); i++)
      ret[i] = CanDecode(in[i], opts[i]);
    return ret;
  }

  /**
   * @brief To be overriden by a CPU codec implementation
   */
  DecodeResult Decode(SampleView<CPUBackend> out, ImageSource *in, DecodeParams opts) override {
    throw std::logic_error("Backend not supported");
  }

  std::vector<DecodeResult> Decode(span<SampleView<CPUBackend>> out,
                                   cspan<ImageSource *> in, cspan<DecodeParams> opts) override {
    assert(out.size() == in.size());
    assert(out.size() == opts.size() || opts.size() == 1);
    std::vector<DecodeResult> ret(out.size());
    for (int i = 0 ; i < in.size(); i++)
      ret[i] = Decode(out[i], in[i], opts.size() > 1 ? opts[i] : opts[0]);
    return ret;
  }

  /**
   * @brief To be overriden by a GPU/mixed codec implementation
   */
  DecodeResult Decode(SampleView<GPUBackend> out, ImageSource *in, DecodeParams opts) override {
    throw std::logic_error("Backend not supported");
  }

  std::vector<DecodeResult> Decode(span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in, cspan<DecodeParams> opts) override {
    assert(out.size() == in.size());
    assert(out.size() == opts.size() || opts.size() == 1);
    std::vector<DecodeResult> ret(out.size());
    for (int i = 0 ; i < in.size(); i++)
      ret[i] = Decode(out[i], in[i], opts.size() > 1 ? opts[i] : opts[0]);
    return ret;
  }

  void SetParam(const char*, const any &) override {}
  any GetParam(const char *key) const override {
    return {};
  }

 protected:
  int device_id_;
  ThreadPool *tp_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_DECODER_IMPL_H_
