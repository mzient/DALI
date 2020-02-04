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

#include "dali/pipeline/data/buffer.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

template <>
double Buffer<CPUBackend>::growth_factor_ = 1.0;

template <>
double Buffer<GPUBackend>::growth_factor_ = 1.0;

template <>
double Buffer<CPUBackend>::shrink_threshold_ = 0.9;

template <>
double Buffer<GPUBackend>::shrink_threshold_ = 0.0;

}  // namespace dali
