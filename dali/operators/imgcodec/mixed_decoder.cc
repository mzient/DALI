// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/imgcodec/image_decoder.h"
#include "dali/operators/imgcodec/roi_image_decoder.h"

namespace dali {
namespace imgcodec {

class MixedDecoder : public ImageDecoder<MixedBackend> {
 public:
  ~MixedDecoder() override = default;
  explicit MixedDecoder(const OpSpec &spec) : ImageDecoder<MixedBackend>(spec) {}

  virtual void SetupCropParams(Workspace &ws) {}

  void RunImpl(Workspace &ws) override {
    SetupCropParams(ws);
    RunImplImpl(ws);
  }
};

using MixedDecoderCrop = WithCropAttr<MixedDecoder, MixedBackend>;
using MixedDecoderSlice = WithSliceAttr<MixedDecoder, MixedBackend>;
using MixedDecoderRandomCrop = WithRandomCropAttr<MixedDecoder, MixedBackend>;

DALI_REGISTER_OPERATOR(decoders__Image, MixedDecoder, Mixed);
DALI_REGISTER_OPERATOR(decoders__ImageCrop, MixedDecoderCrop, Mixed);
DALI_REGISTER_OPERATOR(decoders__ImageSlice, MixedDecoderSlice, Mixed);
DALI_REGISTER_OPERATOR(decoders__ImageRandomCrop, MixedDecoderRandomCrop, Mixed);

// Deprecated aliases: fn.image*_decoder
DALI_REGISTER_OPERATOR(ImageDecoder, MixedDecoder, Mixed);
DALI_REGISTER_OPERATOR(ImageDecoderCrop, MixedDecoderCrop, Mixed);
DALI_REGISTER_OPERATOR(ImageDecoderSlice, MixedDecoderSlice, Mixed);
DALI_REGISTER_OPERATOR(ImageDecoderRandomCrop, MixedDecoderRandomCrop, Mixed);

// Deprecated aliases: fn.experimental.decoders.image*
DALI_REGISTER_OPERATOR(experimental__decoders__Image, MixedDecoder, Mixed);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageCrop, MixedDecoderCrop, Mixed);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageSlice, MixedDecoderSlice, Mixed);
DALI_REGISTER_OPERATOR(experimental__decoders__ImageRandomCrop, MixedDecoderRandomCrop, Mixed);

}  // namespace imgcodec
}  // namespace dali
