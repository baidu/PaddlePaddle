/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <boost/variant.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/framework/attr_checker.h"
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/scope.h"
#include "paddle/platform/device_context.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace framework {

using paddle::platform::DeviceContext;
class OperatorBase;

/**
 * OpRunContext is the only parameter of Operator's Run function.
 * Run will get input/output variables, state such as momentum and
 * device resource such as CUDA stream, cublas handle, etc. from
 * OpRunContext. User should construct it before run the Operator.
 */
class OpContext {
 public:
  OpContext(const OperatorBase* op, std::shared_ptr<Scope> scope,
            DeviceContext* device_context)
      : op(op), scope(scope), device_context(device_context) {}
  ~OpContext() {}

  const Variable* Input(int index) const;
  Variable* Output(int index) const;

 public:
  const OperatorBase* op;
  std::shared_ptr<Scope> scope;
  DeviceContext* device_context;
};

/**
 * OperatorBase has the basic element that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
  virtual ~OperatorBase() {}

  template <typename T>
  inline const T GetAttr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

  std::string DebugString() const;

  /// InferShape infer the size of Variables used by this Operator with
  /// information inside scope
  virtual void InferShape(std::shared_ptr<Scope> scope) const;

  /// Net will call this function to Run an op.
  void Run(std::shared_ptr<Scope> scope, DeviceContext* dev_ctx) {
    OpContext op_ctx(this, scope, dev_ctx);
    Run(&op_ctx);
  }

  /// when implement an Op, your should implement this function.
  /// this function should be moved to OpKernel later
  virtual void Run(OpContext* context) const = 0;

 public:
  OpDesc desc_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
};

}  // namespace framework
}  // namespace paddle
