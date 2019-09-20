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

#ifndef DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
#define DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_

#include <unordered_map>
#include <utility>
#include "dali/pipeline/operators/operator.h"
#include "dali/core/small_vector.h"

namespace dali {

enum class NodeType {
  Function,
  Constant,
  Input
};

struct ExprNode {
  ExprNode *next = nullptr;
  virtual ~ExprNode() = default;
  virtual NodeType type() const noexcept = 0;
  virtual ExprNode *Clone() const = 0;
};

struct FuncExprNode : ExprNode {
  NodeType type() const noexcept override { return NodeType::Function; }
  FuncExprNode *Clone() const override { return new FuncExprNode(*this); }
  std::string function;
  SmallVector<ExprNode *, 3> args;
};

struct InputExprNode : ExprNode {
  NodeType type() const noexcept override { return NodeType::Input; }
  InputExprNode *Clone() const override { return new InputExprNode(*this); }
  int index;
};

struct ConstantExprNode : ExprNode {
  NodeType type() const noexcept override { return NodeType::Input; }
  ConstantExprNode *Clone() const override { return new ConstantExprNode(*this); }
  double value;
};

struct Expression {
  ExprNode *root = nullptr;
  ExprNode *head = nullptr;
  ~Expression() {
    clear();
  }

  void clear() {
    root = nullptr;
    while (head) {
      ExprNode *n = head;
      head = head->next;
      delete n;
    }
  }

  Expression(const Expression &e) {
    *this = e;
  }
  Expression(Expression &&e) {
    root = e.root;
    head = e.head;
    e.root = nullptr;
    e.head = nullptr;
  }
  Expression &operator=(Expression &&e) {
    std::swap(root, e.root);
    std::swap(head, e.head);
    return *this;
  }
  Expression &operator=(const Expression &e) {
    if (&e == this)
      return *this;
    std::unordered_map<ExprNode *, ExprNode *> mapping;
    clear();
    ExprNode **ptail = &head;
    for (ExprNode *n = e.head; n; n = n->next) {
      *ptail = n->Clone();
      if (n == e.root)
        root = *ptail;
      mapping[n] = *ptail;
      ptail = &(*ptail)->next;
    }
    for (ExprNode *n = head; n; n = n->next) {
      if (n->type() == NodeType::Function) {
        auto &args = static_cast<FuncExprNode *>(n)->args;
        for (auto *&arg : args)
          arg = mapping.find(arg)->second;
      }
    }
    return *this;
  }
};

template <typename Backend>
class ArithExprOp : public Operator<Backend> {
 public:

  Expression expr;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_FUSED_CROP_MIRROR_NORMALIZE_H_
