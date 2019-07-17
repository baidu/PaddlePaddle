/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

// StepScopes manages scopes inside RNN.
//    StepScopes::CurScope() get the current scope
//    StepScopes::ExScope() get the ex-scope, or scope in previous time step.
//    StepScopes::Next() move to next time step.
//
// if is_train = False, then
//   there are two scopes for the RNN and just support forward.
// else
//   the len(scopes) == seq_len
//
// if is_backward = True, then
//   reversely access scopes
// else
//   access scopes from begin to end.
class StepScopes {
 public:
  StepScopes(const platform::DeviceContext &dev_ctx,
             const framework::Scope &parent,
             std::vector<framework::Scope *> *scopes, bool is_train,
             size_t seq_len, bool is_backward = false);

  framework::Scope &CurScope();

  framework::Scope &ExScope();

  void Next();

 private:
  framework::Scope &GetScope(size_t scope_id) const;

  size_t counter_;
  std::vector<framework::Scope *> *scopes_;
  bool is_train_;
  bool is_backward_;
};

// Base class for RecurrentOp/RecurrentGradOp
//    Some common protected functions for RecurrentOp/RecurrentGradOp
class RecurrentBase : public framework::OperatorBase {
 public:
  static constexpr char kInputs[] = "inputs";
  static constexpr char kInitialStates[] = "initial_states";
  static constexpr char kParameters[] = "parameters";
  static constexpr char kOutputs[] = "outputs";
  static constexpr char kStepScopes[] = "step_scopes";
  static constexpr char kHasStates[] = "has_states";
  static constexpr char kExStates[] = "ex_states";
  static constexpr char kStates[] = "states";
  static constexpr char kStepBlock[] = "sub_block";
  static constexpr char kReverse[] = "reverse";
  static constexpr char kIsTrain[] = "is_train";
  static constexpr char kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";
#define GRAD_SUFFIX "@GRAD"
  static constexpr char kInputGrads[] = "inputs" GRAD_SUFFIX;
  static constexpr char kOutputGrads[] = "outputs" GRAD_SUFFIX;
  static constexpr char kParamGrads[] = "parameters" GRAD_SUFFIX;
  static constexpr char kInitStateGrads[] = "initial_states" GRAD_SUFFIX;

  RecurrentBase(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs);

 protected:
  // Get SequenceLength from Scope
  //   The sequence length is got from input tensor. The input tensor's
  //   dimension should be [SEQ_LEN, ..., ...]. The first of the tensor's shape
  //   is SEQ_LEN. The second of the tensor's shape could be the batch size or
  //   nested sequence length.
  int64_t GetSequenceLength(const framework::Scope &scope) const;

  // for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
  //                                   map(dst_scope.Var, dst_vars)):
  //   dst_tensor.ShareDataWith(src_tensor)
  static void LinkTensor(const framework::Scope &src_scope,
                         const std::vector<std::string> &src_vars,
                         framework::Scope *dst_scope,
                         const std::vector<std::string> &dst_vars);

  // for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
  //                                   map(dst_scope.Var, dst_vars)):
  //   callback(src_tensor, &dst_tensor)
  template <typename Callback>
  static void LinkTensorWithCallback(const framework::Scope &src_scope,
                                     const std::vector<std::string> &src_vars,
                                     framework::Scope *dst_scope,
                                     const std::vector<std::string> &dst_vars,
                                     Callback callback,
                                     bool is_backward = false);

  // for src_tensor, dst_tensor in zip(map(src_scope.FindVar, src_vars),
  //                                   map(dst_scope.FindVar, dst_vars)):
  //   callback(src_tensor, &dst_tensor)
  template <typename Callback>
  static void LinkTensorWithCallback(const framework::Scope &src_scope,
                                     const std::vector<std::string> &src_vars,
                                     const framework::Scope &dst_scope,
                                     const std::vector<std::string> &dst_vars,
                                     Callback callback,
                                     bool is_backward = false);

  // (seq_len, shape) -> return [seq_len] + list(shape)
  static framework::DDim PrependDims(size_t seq_len,
                                     const framework::DDim &src);

 private:
  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::string &src_var_name,
                           framework::Scope *dst_scope,
                           const std::string &dst_var_name, Callback callback,
                           bool is_backward = false);

  template <typename Callback>
  static void AccessTensor(const framework::Scope &src_scope,
                           const std::string &src_var_name,
                           const framework::Scope &dst_scope,
                           const std::string &dst_var_name, Callback callback,
                           bool is_backward = false);
};

class RecurrentOp : public RecurrentBase {
 public:
  RecurrentOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs);

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override;

 private:
  StepScopes CreateStepScopes(const platform::DeviceContext &dev_ctx,
                              const framework::Scope &scope,
                              size_t seq_len) const;
};

class RecurrentGradOp : public RecurrentBase {
 public:
  RecurrentGradOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs);

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override;

  StepScopes CreateStepScopes(const platform::DeviceContext &dev_ctx,
                              const framework::Scope &scope,
                              size_t seq_len) const;

  std::unordered_set<std::string> List2Set(
      const std::vector<std::string> &list) const;

  std::unordered_set<std::string> LocalVarNames(
      const framework::Scope &scope) const;

  static std::vector<std::string> GradVarLists(
      const std::vector<std::string> &var_names);
};

class RecurrentOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class RecurrentGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDesc> Apply() const;
};

class RecurrentGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override;
};

}  // namespace operators
}  // namespace paddle
