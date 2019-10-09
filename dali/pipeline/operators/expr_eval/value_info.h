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

#ifndef DALI_PIPELINE_OPERATORS_EXPR_EVAL_VALUE_INFO_H_
#define DALI_PIPELINE_OPERATORS_EXPR_EVAL_VALUE_INFO_H_

#include <string>
#include "dali/core/span.h"
#include "dali/core/any.h"
#include "dali/pipeline/data/types.h"
#include "dali/kernels/tensor_shape.h"

namespace dali {
namespace expr {

enum class ValueKind : int {
  Undefined = 0,
  Tensor,
  Scalar,
  Constant
};

struct ArgInfo {
  ValueKind kind = ValueKind::Undefined;
  DALIDataType type = DALI_NO_TYPE;
  int bits = 0;
};

struct ValueInfo : ArgInfo {
  kernels::TensorListShape<> shape;
  any constant;
};

}  // namespace expr
}  // namespace dali


#endif  // DALI_PIPELINE_OPERATORS_EXPR_EVAL_VALUE_INFO_H_
