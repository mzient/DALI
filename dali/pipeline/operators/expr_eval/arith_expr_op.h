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

#ifndef DALI_PIPELINE_OPERATORS_EXPR_EVAL_ARITH_EXPR_OP_H_
#define DALI_PIPELINE_OPERATORS_EXPR_EVAL_ARITH_EXPR_OP_H_

#include <unordered_map>
#include <utility>
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/expr_eval/expr.h"

namespace dali {

template <typename Backend>
class ArithExprOp : public Operator<Backend> {
 public:

  expr::Expression expression;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPR_EVAL_ARITH_EXPR_OP_H_
