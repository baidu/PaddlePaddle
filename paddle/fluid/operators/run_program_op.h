/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using ProgramDesc = framework::ProgramDesc;
using BlockDesc = framework::BlockDesc;

using Variable = framework::Variable;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

using FeedFetchList = framework::FeedFetchList;

namespace details {

static std::string GetSkipEagerDeletionVarsDebugString(
    const std::vector<std::string> &vars) {
  std::string str = "Skip " + std::to_string(vars.size()) +
                    " var(s) in eager deletion mode: ";
  for (auto &var : vars) {
    str.append(var);
    str.push_back(' ');
  }
  return str;
}

static void VariableShare(Variable *src_var, Variable *dst_var) {
  // The previous check ensures that the variable type can only be LoDTensor or
  // SelectedRows
  if (src_var->IsType<LoDTensor>()) {
    auto *lod_tensor = dst_var->GetMutable<LoDTensor>();
    lod_tensor->ShareDataWith(src_var->Get<LoDTensor>());
    lod_tensor->set_lod(src_var->Get<LoDTensor>().lod());
  } else if (src_var->IsType<SelectedRows>()) {
    auto *selected_rows = dst_var->GetMutable<SelectedRows>();
    selected_rows->mutable_value()->ShareDataWith(
        src_var->Get<SelectedRows>().value());
    selected_rows->set_rows(src_var->Get<SelectedRows>().rows());
    selected_rows->set_height(src_var->Get<SelectedRows>().height());
  }
}

static void VariableCopy(Variable *src_var, const platform::Place &dst_place,
                         Variable *dst_var) {
  // The previous check ensures that the variable type can only be LoDTensor or
  // SelectedRows
  if (src_var->IsType<LoDTensor>()) {
    auto *lod_tensor = dst_var->GetMutable<LoDTensor>();
    TensorCopySync(src_var->Get<LoDTensor>(), dst_place, lod_tensor);
    lod_tensor->set_lod(src_var->Get<LoDTensor>().lod());
  } else if (src_var->IsType<SelectedRows>()) {
    auto *selected_rows = dst_var->GetMutable<SelectedRows>();
    TensorCopySync(src_var->Get<SelectedRows>().value(), dst_place,
                   selected_rows->mutable_value());
    selected_rows->set_rows(src_var->Get<SelectedRows>().rows());
    selected_rows->set_height(src_var->Get<SelectedRows>().height());
  }
}

static void ShareVarsIntoScope(const std::vector<Variable *> vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    auto *var = scope.Var(var_names[i]);
    if (vars[i]->IsType<LoDTensor>()) {
      PADDLE_ENFORCE_EQ(
          vars[i]->Get<LoDTensor>().IsInitialized(), true,
          platform::errors::InvalidArgument(
              "The tensor in input variable %s of "
              "RunProgram(Grad)Op(StaticModelRunner) is not initialized.",
              var_names[i]));
      VariableShare(vars[i], var);
    } else if (vars[i]->IsType<SelectedRows>()) {
      PADDLE_ENFORCE_EQ(
          vars[i]->Get<SelectedRows>().value().IsInitialized(), true,
          platform::errors::InvalidArgument(
              "The tensor in input variable %s of "
              "RunProgram(Grad)Op(StaticModelRunner) is not initialized.",
              var_names[i]));
      VariableShare(vars[i], var);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The RunProgram(Grad)Op(StaticModelRunner) only support input "
          "variable of type LoDTensor or SelectedRows, "
          "but received variable %s's type is %s",
          var_names[i],
          platform::demangle(framework::ToTypeName(vars[i]->Type()))));
    }
  }
}

static void CheckOutputVarStatus(Variable *src_var, Variable *dst_var,
                                 const std::string &var_name) {
  if (dst_var->IsType<LoDTensor>()) {
    PADDLE_ENFORCE_EQ(
        src_var->IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op(StaticModelRunner)'s internal scope holds "
            "wrong type. Expect type is LoDTensor, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var->Type()))));
    PADDLE_ENFORCE_EQ(src_var->Get<LoDTensor>().IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable % get from "
                          "RunProgram(Grad)Op(StaticModelRunner)'s internal "
                          "scope is not initialized.",
                          var_name));
  } else if (dst_var->IsType<SelectedRows>()) {
    PADDLE_ENFORCE_EQ(
        src_var->IsType<SelectedRows>(), true,
        platform::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op(StaticModelRunner)'s internal scope holds "
            "wrong type. Expect type is SelectedRows, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var->Type()))));
    PADDLE_ENFORCE_EQ(src_var->Get<SelectedRows>().value().IsInitialized(),
                      true, platform::errors::InvalidArgument(
                                "The tensor in output variable % get from "
                                "RunProgram(Grad)Op(StaticModelRunner)'s "
                                "internal scope is not initialized.",
                                var_name));

  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op(StaticModelRunner) only support output "
        "variable of type LoDTensor or SelectedRows, "
        "but received variable %s's type is %s",
        var_name, platform::demangle(framework::ToTypeName(dst_var->Type()))));
  }
}

static void ShareVarsFromScope(std::vector<Variable *> vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    auto *var = scope.FindVar(var_names[i]);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("The output variable %s is not in "
                                        "RunProgram(Grad)Op(StaticModelRunner)'"
                                        "s internal scope.",
                                        var_names[i]));
    CheckOutputVarStatus(var, vars[i], var_names[i]);
    VariableShare(var, vars[i]);
  }
}

static void CopyVarsFromScope(std::vector<Variable *> vars,
                              const std::vector<std::string> &var_names,
                              const platform::Place &dst_place,
                              framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    auto *var = scope.FindVar(var_names[i]);
    if (nullptr == var) {
      // need remove not find output?
      VLOG(2) << "Can't find variable " << var_names[i]
              << "in RunProgram(Grad)Op(StaticModelRunner)'s internal scope.";
      continue;
    } else {
      CheckOutputVarStatus(var, vars[i], var_names[i]);
      VariableCopy(var, dst_place, vars[i]);
    }
  }
}
}  // namespace details

template <typename DeviceContext, typename T>
class RunProgramOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramOpKernel Compute";
    // Step 1. prepare inputs, outputs, attrs
    auto &input_vars = ctx.MultiInputVar("X");
    auto &param_vars = ctx.MultiInputVar("Params");
    auto output_vars = ctx.MultiOutputVar("Out");

    auto input_var_names = ctx.InputNames("X");
    auto param_names = ctx.InputNames("Params");
    auto output_var_names = ctx.OutputNames("Out");

    auto *block = ctx.Attr<BlockDesc *>("fwd_block");
    auto *fwd_program = block->Program();

    // NOTE(chenweihang): In order not to add new variable type, use vector
    // here.
    // Originally, here can use scope directly.
    auto *out_scope_vec = ctx.Output<StepScopeVar>("OutScope");

    // TODO(chenweihang): check input output size
    PADDLE_ENFORCE_EQ(out_scope_vec->size(), 0,
                      "The StepScope should be empty.");

    // Step 2. prepare executor and init persistable variables
    framework::Executor exe(ctx.GetPlace());
    // framework::Scope scope;
    out_scope_vec->emplace_back(new framework::Scope());
    framework::Scope &scope = *(out_scope_vec->front());

    // share input_vars & parameters into scope
    details::ShareVarsIntoScope(input_vars, input_var_names, &scope);
    details::ShareVarsIntoScope(param_vars, param_names, &scope);

    // Step 3. run ops
    exe.Run(*fwd_program, &scope, 0, false, true, {}, true, true);

    // Step 4. Get Output
    details::ShareVarsFromScope(output_vars, output_var_names, &scope);

    // Debug info: scope info when run end
    VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());
  }
};

template <typename DeviceContext, typename T>
class RunProgramGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(2) << "RunProgramGradOpKernel Compute";
    // Step 1. prepare inputs and outputs
    auto &output_grad_vars = ctx.MultiInputVar(framework::GradVarName("Out"));
    auto input_grad_vars = ctx.MultiOutputVar(framework::GradVarName("X"));
    auto param_grad_vars = ctx.MultiOutputVar(framework::GradVarName("Params"));

    auto output_grad_var_names = ctx.InputNames(framework::GradVarName("Out"));
    auto input_grad_var_names = ctx.OutputNames(framework::GradVarName("X"));
    auto param_grad_names = ctx.OutputNames(framework::GradVarName("Params"));
    // remove prefix Gtmp@
    auto rm_prefix_func = [](std::string &name) {
      name = std::move(name.substr(5));
    };
    std::for_each(input_grad_var_names.begin(), input_grad_var_names.end(),
                  rm_prefix_func);
    std::for_each(param_grad_names.begin(), param_grad_names.end(),
                  rm_prefix_func);

    std::stringstream ss;
    ss << "Maker names: ";
    for (size_t i = 0; i < input_grad_var_names.size(); ++i) {
      ss << input_grad_var_names[i] << ", ";
    }
    for (size_t i = 0; i < param_grad_names.size(); ++i) {
      ss << param_grad_names[i] << ", ";
    }
    VLOG(2) << ss.str();

    auto output_names = ctx.OutputNames(framework::GradVarName("Params"));
    std::stringstream ss2;
    ss2 << "Output names: ";
    for (size_t i = 0; i < output_names.size(); ++i) {
      ss2 << output_names[i] << ", ";
    }
    VLOG(2) << ss2.str();

    auto *block = ctx.Attr<BlockDesc *>("bwd_block");
    auto *bwd_program = block->Program();

    auto *out_scope_vec = ctx.Input<StepScopeVar>("OutScope");
    PADDLE_ENFORCE_EQ(out_scope_vec->size(), 1,
                      "The StepScope should only hold one scope.");

    // Step 2. prepare executor and scope
    framework::Executor exe(ctx.GetPlace());
    auto &scope = *(out_scope_vec->front());

    // skip delete vars, out@grad & params@grad
    /*
    std::vector<std::string> skip_vars;
    std::copy(output_grad_var_names.begin(), output_grad_var_names.end(),
              std::back_inserter(skip_vars));
    std::copy(param_grad_names.begin(), param_grad_names.end(),
              std::back_inserter(skip_vars));
    VLOG(2) << details::GetSkipEagerDeletionVarsDebugString(skip_vars);
    */

    details::ShareVarsIntoScope(output_grad_vars, output_grad_var_names,
                                &scope);

    // Debug info: scope info when run end
    VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());

    // Step 3. run ops
    // exe.Run(*bwd_program, &scope, 0, false, true, skip_vars);
    exe.Run(*bwd_program, &scope, 0, false, true, {}, true, true);

    // Step 4. copy outputs
    // TODO(chenweihang): need copy?
    details::CopyVarsFromScope(input_grad_vars, input_grad_var_names,
                               ctx.GetPlace(), &scope);
    details::CopyVarsFromScope(param_grad_vars, param_grad_names,
                               ctx.GetPlace(), &scope);

    // Debug info: scope info when run end
    VLOG(3) << framework::GenScopeTreeDebugInfo(out_scope_vec->front());

    // Step 5. clear
    // scope_vec->clear();
  }
};

}  // namespace operators
}  // namespace paddle
