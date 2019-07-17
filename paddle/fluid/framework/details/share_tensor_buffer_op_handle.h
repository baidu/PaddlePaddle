// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"

namespace paddle {
namespace framework {

class Variable;
class Scope;
class Tensor;

namespace ir {
class MemOptVarInfo;
}  // namespace ir

namespace details {

class ShareTensorBufferFunctor {
 public:
  ShareTensorBufferFunctor(Scope *scope, size_t scope_idx,
                           const std::string &op_type,
                           const std::vector<ir::MemOptVarInfo *> &in_var_infos,
                           const std::vector<std::string> &out_var_names);

  void Add(ir::MemOptVarInfo *in_var_info, const std::string &out_var_name);

  void operator()(Scope *exec_scope);

  std::unordered_map<std::string, std::string> ReusedVars() const;

  size_t GetScopeIdx() const { return scope_idx_; }

  Scope *GetScope() { return scope_; }

 private:
  void CallOnce();

 private:
  Scope *scope_;
  Scope *exec_scope_{nullptr};

  size_t scope_idx_;
  std::string op_type_;
  std::vector<ir::MemOptVarInfo *> in_var_infos_;
  std::vector<std::string> out_var_names_;

  std::vector<std::pair<const Variable *, Variable *>> in_out_vars_;
};

class ShareTensorBufferOpHandle : public OpHandleBase {
 public:
  ShareTensorBufferOpHandle(
      ir::Node *node, Scope *scope, size_t scope_idx,
      const std::string &op_type,
      const std::vector<ir::MemOptVarInfo *> &in_vars_infos,
      const std::vector<std::string> &out_var_names);

  std::unordered_map<std::string, std::string> ReusedVars() const;

  Priority GetPriority() const override { return Priority::kHighest; }

  size_t GetScopeIdx() const { return functor_.GetScopeIdx(); }

  void Add(ir::MemOptVarInfo *in_var_info, const std::string &ou_var_name);

  const ShareTensorBufferFunctor &Functor() const { return functor_; }

 protected:
  std::string Name() const override { return "buffer_share"; }

  void RunImpl() final;

  void InitCUDA() override;

  std::vector<Scope *> GetLocalScopes() override {
    return {functor_.GetScope()};
  }

 private:
  ShareTensorBufferFunctor functor_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
