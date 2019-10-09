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


#ifndef DALI_PIPELINE_OPERATORS_EXPR_EVAL_FUNCTION_H_
#define DALI_PIPELINE_OPERATORS_EXPR_EVAL_FUNCTION_H_

#include <string>
#include "dali/kernels/alloc.h"
#include "dali/core/span.h"
#include "dali/core/any.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/pipeline/operators/expr_eval/value_info.h"

namespace dali {
namespace expr {

template <typename Backend>
struct ExecContext {
  workspace_t<Backend> *workspace = nullptr;
  std::vector<any> constants;

};

struct Tile {
  void *data;
  int element_size;
  int num_elements;
};

template <typename Backend>
class FunctionImpl {
 public:
  virtual void Run(span<const Tile> arg_tiles) = 0;
  virtual void SetScalarArgs(span<const any> args) = 0;
  virtual span<const ArgInfo> Args() = 0;
  virtual ArgInfo RetInfo() = 0;
  virtual ~FunctionImpl() = default;
};

class CallInfo {
 public:
  virtual void SetArgInfo(const char *function, span<const ValueInfo> args) = 0;
  virtual ValueInfo OutputInfo() const = 0;
  virtual int Arity() const = 0;
  virtual ~CallInfo() = default;
};

class FunctionTable {
 public:
  FunctionTable &instance();

  FunctionImpl<GPUBackend> *GetImplGPU(const CallInfo &info) const;
  FunctionImpl<CPUBackend> *GetImplCPU(const CallInfo &info) const;
  void Register(std::unique_ptr<FunctionImpl<GPUBackend>> impl_gpu);
  void Register(std::unique_ptr<FunctionImpl<CPUBackend>> impl_cpu);
  template <typename Impl>
  void Register(const Impl &impl) {
    Register(std::make_unique<Impl>(impl));
  }
};

}  // namespace expr
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_EXPR_EVAL_FUNCTION_H_
