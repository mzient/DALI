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

#ifndef DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_H_
#define DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_H_

#include <memory>
#include "dali/core/span.h"

namespace dali {

struct AudioMetadata {
  int length;
  int sample_rate;  /// [Hz]
  int channels;
  bool channels_interleaved;
};

class AudioDecoderBase {
 public:
  AudioMetadata Open(span<const char> encoded) {
    Close();
    return OpenImpl(encoded);
  }


  void Close() {
    CloseImpl();
  }


  virtual void Decode(span<char> raw_output) = 0;

  virtual ~AudioDecoderBase() = default;

 private:
  virtual AudioMetadata OpenImpl(span<const char> encoded) = 0;

  virtual void CloseImpl() = 0;
};

template<typename SampleType>
class TypedAudioDecoderBase : public AudioDecoderBase {
 public:
  void Decode(span<char> raw_output) override {
    int max_samples = static_cast<int>(raw_output.size() / sizeof(SampleType));
    DecodeTyped({reinterpret_cast<SampleType *>(raw_output.data()), max_samples});
  }


  virtual void DecodeTyped(span<SampleType> typed_output) = 0;
};




}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_H_
