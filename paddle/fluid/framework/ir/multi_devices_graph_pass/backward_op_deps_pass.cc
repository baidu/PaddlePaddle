// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

class BackWardOpDepsPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    // NOTE: The operator nodes should be in topology order.
    std::vector<details::OpHandleBase*> backward_op_handles;
    std::vector<details::OpHandleBase*> opt_handles;
    std::vector<ir::Node*> topo_nodes = ir::TopologySortOperations(*graph);
    for (auto& node : topo_nodes) {
      GetBackWardOpHandles(node, &backward_op_handles);
      GetOptHandles(node, &opt_handles);
    }

    if (backward_op_handles.size() <= 1 || opt_handles.size() <= 1) {
      VLOG(10) << string::Sprintf(
          "backward_op_handles size:%d, opt_handles:%d, so need not to add "
          "deps between them",
          static_cast<int>(backward_op_handles.size()),
          static_cast<int>(opt_handles.size()));
      return;
    }

    for (size_t i = 1; i < backward_op_handles.size(); ++i) {
      auto* dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
      graph->Get<details::GraphDepVars>(details::kGraphDepVars)
          .emplace(dep_var);
      backward_op_handles[i - 1]->AddOutput(dep_var);
      backward_op_handles[i]->AddInput(dep_var);
    }

    for (size_t i = 1; i < opt_handles.size(); ++i) {
      auto* dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
      graph->Get<details::GraphDepVars>(details::kGraphDepVars)
          .emplace(dep_var);
      opt_handles[i - 1]->AddOutput(dep_var);
      opt_handles[i]->AddInput(dep_var);
    }

    auto* dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<details::GraphDepVars>(details::kGraphDepVars).emplace(dep_var);
    backward_op_handles[backward_op_handles.size() - 1]->AddOutput(dep_var);
    opt_handles[0]->AddInput(dep_var);
  }

  void GetBackWardOpHandles(
      ir::Node* node,
      std::vector<details::OpHandleBase*>* backward_op_handles) const {
    try {
      bool is_bk_op =
          static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                OpProtoAndCheckerMaker::OpRoleAttrName())) &
                            static_cast<int>(OpRole::kBackward));
      if (!is_bk_op) return;

      // Currently, we assume that once gradient is generated, it can be
      // broadcast, and each gradient is only broadcast once.
      auto backward_vars =
          boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
              OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      PADDLE_ENFORCE_EQ(backward_vars.size() % 2, static_cast<size_t>(0));
      PADDLE_ENFORCE(node->IsWrappedBy<details::OpHandleBase>());

      backward_op_handles->emplace_back(
          &node->Wrapper<details::OpHandleBase>());
    } catch (boost::bad_get e) {
    }
  }

  void GetOptHandles(ir::Node* node,
                     std::vector<details::OpHandleBase*>* opt_handles) const {
    try {
      bool is_opt_op =
          static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                OpProtoAndCheckerMaker::OpRoleAttrName())) &
                            static_cast<int>(OpRole::kOptimize));
      if (!is_opt_op) return;

      opt_handles->emplace_back(&node->Wrapper<details::OpHandleBase>());
    } catch (boost::bad_get e) {
    }
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(backward_op_deps_pass, paddle::framework::ir::BackWardOpDepsPass);
