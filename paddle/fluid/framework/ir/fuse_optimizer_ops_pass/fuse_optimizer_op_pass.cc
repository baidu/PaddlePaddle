//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_optimizer_ops_pass/fuse_optimizer_op_pass.h"
#include <algorithm>
#include <unordered_set>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseOptimizerOpPass::ApplyImpl(ir::Graph *graph) const {
  ir::Graph &result = *graph;

  const std::string fuse_op_type = GetOpType();
  std::vector<std::string> aux_var_names = GetAuxiliaryVarNames();
  aux_var_names.emplace_back(kParam);
  aux_var_names.emplace_back(kGrad);

  // Step 1: Get the specified op and auxiliary variables.
  std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(result);
  auto vars_info = GetVarInfo(result);
  std::vector<ir::Node *> opt_nodes;
  size_t opt_ops_num = 0;
  // Note: Only take care about the dense gradients.
  for (auto &node : topo_nodes) {
    if (node->Op()->Type() == fuse_op_type) {
      auto grad_name = node->Op()->Input(kGrad);
      PADDLE_ENFORCE_EQ(grad_name.size(), static_cast<size_t>(1));
      if (GettypeOfVar(vars_info, grad_name[0]) == proto::VarType::LOD_TENSOR) {
        opt_nodes.emplace_back(node);
      }
      ++opt_ops_num;
    }
  }

  VLOG(6) << "Find " << fuse_op_type << " operators : " << opt_ops_num
          << ", and " << opt_nodes.size() << " for dense gradients ";
  if (opt_nodes.size() == 0 || result.Has(details::kFusedOptType)) {
    if (result.Has(details::kFusedOptType)) {
      auto &opt_type =
          result.Get<details::FusedOptType>(details::kFusedOptType);
      VLOG(6) << "Currently only support fusing one type optimizer op. "
                 "Has fused "
              << opt_type;
    }
    return;
  }
  result.Set(details::kFusedOptType, new details::FusedOptType);
  result.Get<details::FusedOptType>(details::kFusedOptType) = fuse_op_type;

  // Step 2: Insert fused_var_name to FusedVars, and the FusedVars need be
  // initialized in scopes before execution.
  if (!result.Has(details::kFusedVars)) {
    result.Set(details::kFusedVars, new details::FusedVars);
  }
  std::unordered_map<std::string, std::vector<std::string>> aux_var_set;
  GetSpecifiedOpsAndVars(aux_var_names, opt_nodes, &aux_var_set);
  std::unordered_map<std::string, std::string> fused_vars_name;
  fused_vars_name.reserve(aux_var_names.size());
  auto &fused_var_set = result.Get<details::FusedVars>(details::kFusedVars);
  const std::string prefix(details::kFusedVarNamePrefix);
  for (auto &var_name : aux_var_names) {
    // NOTE: the fused_var_name should be unique.
    auto fused_var_name = prefix + "_" + fuse_op_type + "_" + var_name + "_" +
                          aux_var_set[var_name][0];
    VLOG(6) << var_name << ": " << fused_var_name;
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0);
    fused_var_set.insert(fused_var_name);
    fused_vars_name.emplace(var_name, fused_var_name);
  }

  // Step 3: Get the fused Gradient's name
  bool grad_fused = false;
  if (result.Has(details::kParamsAndDenseGrads)) {
    // NOTE: kParamsAndDenseGrads is generated by
    // alloc_continue_space_for_grad_pass
    auto &params_and_dense_grads =
        result.Get<details::ParamsAndGrads>(details::kParamsAndDenseGrads);
    PADDLE_ENFORCE_EQ(
        params_and_dense_grads.size(), aux_var_set.at(kGrad).size(),
        "The number of gradients and optimizer ops is not equal.");
    std::unordered_set<std::string> opt_grad_set(aux_var_set.at(kGrad).size());
    for (auto &p_g : params_and_dense_grads) {
      opt_grad_set.insert(p_g.second);
    }
    std::vector<size_t> new_grad_idx;
    for (size_t idx = 0; idx < aux_var_set.at(kGrad).size(); ++idx) {
      auto &grad = aux_var_set.at(kGrad).at(idx);
      if (!opt_grad_set.count(grad)) {
        new_grad_idx.emplace_back(idx);
      }
    }

    // NOTE(zcd): the gradient of kParamsAndDenseGrads may be different
    // with the kGrad. The gradients of kParamsAndDenseGrads is
    // collected during backward stage, but in optimization state, the
    // some gradient's name maybe changed.
    if (new_grad_idx.size() == 0) {
      if (!result.Has(details::kFusedGrads)) {
        PADDLE_THROW(
            "The alloc_continuous_space_for_grad_pass should "
            "be called before this pass.");
      }
      auto &fused_grad = result.Get<details::FusedGrads>(details::kFusedGrads);
      PADDLE_ENFORCE_NE(fused_grad.size(), 0,
                        "The fused gradient should not be empty.");
      PADDLE_ENFORCE_EQ(fused_grad.size(), 1,
                        "Because the dtype of those gradients "
                        "is not unified, so the number of fused gradients is "
                        "more than one, but it is not supported currently.");
      auto &fused_vars = result.Get<details::FusedVars>(details::kFusedVars);
      auto iter =
          std::find(fused_vars.begin(), fused_vars.end(), fused_grad.front());
      PADDLE_ENFORCE(iter != fused_vars.end(), "Not find the fused_grad.");
      fused_vars_name[kGrad] = fused_grad.front();

      // Sort the parameters and auxiliary variables according
      // to parameters' name to make variables' name correspond correctly.
      SortParametersAndAuxVars(params_and_dense_grads, &aux_var_set,
                               &opt_nodes);
      grad_fused = true;
    } else {
      if (new_grad_idx.size() == 1) return;
      // NOTE(zcd): If the gradients of backward stage and optimization stage
      // have diff, Only take care of the the gradient of optimization stage.
      GradientsFilter(new_grad_idx, &opt_nodes, &aux_var_set);
    }
  }

  // Step 4: Alloc continuous space for Parameters and AuxiliaryVar(e.g.
  // Moment1, Moment2, Beta1Pow, Beta2Pow) of all the optimizer ops
  // separately.
  auto &places = Get<const std::vector<platform::Place>>(details::kPlaces);
  auto &local_scopes = Get<const std::vector<Scope *>>(details::kLocalScopes);
  if (!grad_fused) {
    InitFusedGradsAndAllocSpaceForGrads(
        places, local_scopes, aux_var_set.at(kParam), aux_var_set.at(kGrad),
        fused_vars_name.at(kGrad), &result);
  }
  aux_var_names.pop_back();
  InitFusedVarsAndAllocSpaceForVars(places, local_scopes, aux_var_names,
                                    aux_var_set, fused_vars_name);

  // Step 5: Fuse optimizer Ops and Scale Ops
  FuseOptimizerOps(aux_var_set, fused_vars_name, opt_nodes, &result);

  // Step 6: Remove optimizer Ops
  for (auto &opt_op : opt_nodes) {
    graph->RemoveNode(opt_op);
  }
}

void FuseOptimizerOpPass::GradientsFilter(
    const std::vector<size_t> &new_grad_idx, std::vector<Node *> *opt_nodes,
    std::unordered_map<std::string, std::vector<std::string>> *aux_var_set)
    const {
  for (auto &aux_vars : *aux_var_set) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i : new_grad_idx) {
      sorted_vars.emplace_back(aux_vars.second.at(i));
    }
    std::swap(aux_vars.second, sorted_vars);
    if (VLOG_IS_ON(6)) {
      std::stringstream out;
      for (auto &var_name : aux_vars.second) {
        out << var_name << " ";
      }
      VLOG(6) << aux_vars.first << ": " << out.str();
    }
  }
  std::vector<Node *> sorted_ops;
  for (size_t i : new_grad_idx) {
    sorted_ops.emplace_back(opt_nodes->at(i));
  }
  std::swap(*opt_nodes, sorted_ops);
}

void FuseOptimizerOpPass::InitFusedGradsAndAllocSpaceForGrads(
    const std::vector<platform::Place> &places,
    const std::vector<Scope *> &local_scopes,
    const std::vector<std::string> &params,
    const std::vector<std::string> &grads, const std::string &fused_grad_name,
    ir::Graph *result) const {
  auto vars_info = GetVarInfo(*result);
  // Set Gradients as Persistable to prevent this var becoming reusable.
  for (auto &grad_var_name : grads) {
    auto iter = vars_info.find(grad_var_name);
    PADDLE_ENFORCE(iter != vars_info.end());
    PADDLE_ENFORCE(!iter->second.empty());
    PADDLE_ENFORCE_NOT_NULL(iter->second.front()->Var());
    PADDLE_ENFORCE(
        iter->second.front()->Var()->GetType() == proto::VarType::LOD_TENSOR,
        "Currently the gradient type only should be LoDTensor when "
        "fusing optimizer ops.");
    for (auto var : iter->second) {
      var->Var()->SetPersistable(true);
    }
  }

  // Init Grads
  for (auto it = local_scopes.rbegin(); it != local_scopes.rend(); ++it) {
    auto &scope = *it;
    VLOG(6) << "Init: " << fused_grad_name;
    PADDLE_ENFORCE(scope->FindVar(fused_grad_name) == nullptr,
                   "%s has existed in scope.", fused_grad_name);
    scope->Var(fused_grad_name)->GetMutable<LoDTensor>();
    for (auto &grad_var_name : grads) {
      auto iter = vars_info.find(grad_var_name);
      PADDLE_ENFORCE(iter != vars_info.end());
      PADDLE_ENFORCE(!iter->second.empty());
      PADDLE_ENFORCE_NOT_NULL(iter->second.front()->Var());
      scope->Var(grad_var_name)->GetMutable<LoDTensor>();
    }
  }
  // Define Ops
  ProgramDesc program_desc;
  auto *global_block = program_desc.MutableBlock(0);
  AppendAllocContinuousSpace(params, grads, fused_grad_name, global_block,
                             false, false);
  // Run Ops
  RunInitOps(places, local_scopes, *global_block);
}

std::unordered_map<std::string, std::vector<Node *>>
FuseOptimizerOpPass::GetVarInfo(const Graph &result) const {
  std::unordered_map<std::string, std::vector<Node *>> vars;
  for (Node *node : result.Nodes()) {
    if (node->IsVar() && node->Var()) {
      // Note: The graph may have the same name node. For example, parameter
      // is the input of operator and it also is the output of optimizer;
      vars[node->Var()->Name()].emplace_back(node);
    }
  }
  return vars;
}

proto::VarType::Type FuseOptimizerOpPass::GettypeOfVar(
    const std::unordered_map<std::string, std::vector<Node *>> &var_nodes,
    const std::string &name) const {
  auto grad_iter = var_nodes.find(name);
  PADDLE_ENFORCE(grad_iter != var_nodes.end());
  PADDLE_ENFORCE(grad_iter->second.size() > 0);
  PADDLE_ENFORCE_NOT_NULL(grad_iter->second.front()->Var());
  return grad_iter->second.front()->Var()->GetType();
}

void FuseOptimizerOpPass::InitFusedVarsAndAllocSpaceForVars(
    const std::vector<platform::Place> &places,
    const std::vector<Scope *> &local_scopes,
    const std::vector<std::string> &aux_var_names,
    const std::unordered_map<std::string, std::vector<std::string>>
        &aux_var_set,
    const std::unordered_map<std::string, std::string> &fused_vars_name) const {
  // Init Vars
  for (auto &var_name : aux_var_names) {
    auto &fused_var_name = fused_vars_name.at(var_name);
    InitVars(local_scopes, fused_var_name);
  }
  // Define Ops
  ProgramDesc program_desc;
  auto *global_block = program_desc.MutableBlock(0);
  for (auto &var_name : aux_var_names) {
    AppendAllocContinuousSpace(
        aux_var_set.at(var_name), aux_var_set.at(var_name),
        fused_vars_name.at(var_name), global_block, true);
  }
  // Run Ops
  RunInitOps(places, local_scopes, *global_block);
}

void FuseOptimizerOpPass::RunInitOps(const std::vector<platform::Place> &places,
                                     const std::vector<Scope *> &local_scopes,
                                     const BlockDesc &global_block) const {
  for (size_t i = 0; i < local_scopes.size(); ++i) {
    for (auto &op_desc : global_block.AllOps()) {
      auto op = OpRegistry::CreateOp(*op_desc);
      op->Run(*local_scopes[i], places[i]);
    }
  }
}

void FuseOptimizerOpPass::InitVars(const std::vector<Scope *> &local_scopes,
                                   const std::string &fused_var_name) const {
  // Alloc parameters and auxiliary vars in the respective scope.
  size_t idx = local_scopes.size();
  for (auto iter = local_scopes.rbegin(); iter != local_scopes.rend();
       ++iter, --idx) {
    auto &scope = *iter;
    VLOG(6) << "Init: " << fused_var_name;
    PADDLE_ENFORCE(scope->FindVar(fused_var_name) == nullptr,
                   "%s has exist in scope[%d]", fused_var_name, idx);
    scope->Var(fused_var_name)->GetMutable<LoDTensor>();
  }
}

void FuseOptimizerOpPass::SortParametersAndAuxVars(
    const std::vector<std::pair<std::string, std::string>> &params_grads,
    std::unordered_map<std::string, std::vector<std::string>> *aux_vars_set,
    std::vector<ir::Node *> *ops) const {
  PADDLE_ENFORCE_NE(aux_vars_set->count(kParam), static_cast<size_t>(0));
  auto &param_vec = aux_vars_set->at(kParam);

  std::vector<size_t> param_sort_idx;
  param_sort_idx.reserve(param_vec.size());

  for (auto &p_g : params_grads) {
    auto iter = std::find(param_vec.begin(), param_vec.end(), p_g.first);
    PADDLE_ENFORCE(iter != param_vec.end());
    auto idx = std::distance(param_vec.begin(), iter);
    param_sort_idx.emplace_back(idx);
  }

  for (auto &aux_vars : *aux_vars_set) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i = 0; i < aux_vars.second.size(); ++i) {
      sorted_vars.emplace_back(aux_vars.second.at(param_sort_idx[i]));
    }
    std::swap(aux_vars.second, sorted_vars);

    if (VLOG_IS_ON(6)) {
      std::stringstream out;
      for (auto &var_name : aux_vars.second) {
        out << var_name << " ";
      }
      VLOG(6) << aux_vars.first << ": " << out.str();
    }
  }

  std::vector<ir::Node *> sorted_ops;
  sorted_ops.reserve(ops->size());
  for (size_t i = 0; i < ops->size(); ++i) {
    sorted_ops.emplace_back(ops->at(param_sort_idx[i]));
  }
  std::swap(*ops, sorted_ops);
}

void FuseOptimizerOpPass::GetSpecifiedOpsAndVars(
    const std::vector<std::string> &aux_vars_name,
    const std::vector<ir::Node *> &opt_nodes,
    std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
    const {
  for (auto &node : opt_nodes) {
    std::stringstream out;
    for (auto &var_n : aux_vars_name) {
      auto arg_names = node->Op()->Input(var_n);
      PADDLE_ENFORCE_EQ(arg_names.size(), static_cast<size_t>(1));
      (*aux_args_name)[var_n].emplace_back(arg_names[0]);
      out << var_n << ", " << arg_names[0] << "; ";
    }
  }
}

void FuseOptimizerOpPass::AppendAllocContinuousSpace(
    const std::vector<std::string> &in_args,
    const std::vector<std::string> &out_args, const std::string &fused_out_arg,
    BlockDesc *global_block, bool copy_data, bool check_name) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("alloc_continuous_space");
  op_desc->SetInput("Input", in_args);
  op_desc->SetOutput("Output", out_args);
  op_desc->SetOutput("FusedOutput", {fused_out_arg});
  op_desc->SetAttr("copy_data", copy_data);
  op_desc->SetAttr("check_name", check_name);
}

void FuseOptimizerOpPass::InserInputAndOutputForOptOps(
    const std::vector<ir::Node *> &opt_nodes, ir::Node *opt_node) const {
  std::unordered_set<ir::Node *> inputs;
  std::unordered_set<ir::Node *> outputs;
  for (auto opt_op : opt_nodes) {
    // set inputs
    inputs.insert(opt_op->inputs.begin(), opt_op->inputs.end());
    for (auto &input : opt_op->inputs) {
      replace(input->outputs.begin(), input->outputs.end(), opt_op, opt_node);
    }
    // set outputs
    outputs.insert(opt_op->outputs.begin(), opt_op->outputs.end());
    for (auto &output : opt_op->outputs) {
      replace(output->inputs.begin(), output->inputs.end(), opt_op, opt_node);
    }
  }
  opt_node->inputs.insert(opt_node->inputs.begin(), inputs.begin(),
                          inputs.end());
  opt_node->outputs.insert(opt_node->outputs.begin(), outputs.begin(),
                           outputs.end());
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
