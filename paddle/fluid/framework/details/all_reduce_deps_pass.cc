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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_deps_pass.h"
#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_graph_view.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace details {

VarHandle* GetValidInput(const AllReduceOpHandle* a) {
  VLOG(11) << "GetValidInput:" << a;
  VLOG(11) << "GetValidInput, a:" << a->DebugString();
  for (auto p : a->Inputs()) {
    VarHandle* b = dynamic_cast<VarHandle*>(p);
    if (b) {
      VLOG(11) << "pointer:" << b;
      return b;
    }
  }

  PADDLE_ENFORCE(false, "not valid allreduce op");
  VLOG(11) << "meets nullptr";
  return nullptr;
}

struct SortNode {
  SortNode(AllReduceOpHandle* op, std::unordered_map<std::string, int>* vars) {
    op_ = op;
    // level_ = level;
    vars_ = vars;
  }

  AllReduceOpHandle* op_;
  // int level_;
  std::unordered_map<std::string, int>* vars_;

  bool operator<(const SortNode& r) const {
    VarHandle* i0 = dynamic_cast<VarHandle*>(GetValidInput(op_));
    VarHandle* i1 = dynamic_cast<VarHandle*>(GetValidInput(r.op_));
    if (i0 == nullptr || i1 == nullptr) {
      VLOG(11) << op_->DebugString();
      VLOG(11) << r.op_->DebugString();
    }

    PADDLE_ENFORCE(i0 != nullptr && i1 != nullptr,
                   "Convert to VarHandle error");

    auto l_it = vars_->find(i0->name_);
    auto r_it = vars_->find(i1->name_);

    if (l_it->second < r_it->second) return true;

    if (l_it->second == r_it->second) {
      return i0->name_ < i1->name_;
    }

    return false;
  }
};

std::unique_ptr<ir::Graph> AllReduceDepsPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& graph_ops = graph->Get<GraphOps>(kGraphOps);

  // get vars level
  int level = 0;
  std::unordered_map<std::string, int> vars;
  auto& ops = Get<const std::vector<OpDesc*>>(kAllOpDescs);
  for (auto* op_desc : ops) {
    auto outputs = op_desc->Outputs();
    for (auto& o_it : outputs) {
      for (auto& v : o_it.second) {  // values
        vars[v] = level;
      }
    }

    level++;
  }

  std::unordered_map<AllReduceOpHandle*, int> allreduce_ops;
  // get allreduce ops.
  for (auto& op : graph_ops) {
    if (op->Name() == "allreduce" || op->Name() == "all_reduce") {
      auto* allreduce_op = dynamic_cast<AllReduceOpHandle*>(op.get());
      PADDLE_ENFORCE(allreduce_op != nullptr, "Convert to allreduce_op error");
      allreduce_ops[allreduce_op] = 0;
    }
  }

  std::vector<SortNode> op_list;
  for (auto& op : allreduce_ops) {
    SortNode n(op.first, &vars);
    op_list.push_back(n);
  }

  std::sort(op_list.begin(), op_list.end());

  // add dependency.
  auto& sorted_ops = op_list;
  for (size_t i = 1; i < sorted_ops.size(); ++i) {
    auto* dep_var = new dummyvarhandle(graph->CreateControlDepVar());

    auto* pre_op = sorted_ops[i - 1].op_;
    auto* op = sorted_ops[i].op_;

    pre_op->AddOutput(dep_var);
    op->AddInput(dep_var);
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);

    VLOG(10) << "add all_reduce sequential dependencies between " << pre_op
             << " and " << op;
  }

  return graph;
}

/*
struct SortNode {
  SortNode(AllReduceOpHandle* op, int level) {
    op_ = op;
    level_ = level;
  }

  AllReduceOpHandle* op_;
  int level_;

  bool operator<(const SortNode& r) const {
    if (level_ < r.level_) return true;

    VarHandle* i0 = dynamic_cast<VarHandle*>(GetValidInput(op_));
    VarHandle* i1 = dynamic_cast<VarHandle*>(GetValidInput(r.op_));
    if (i0 == nullptr || i1 == nullptr) {
      VLOG(11) << op_->DebugString();
      VLOG(11) << r.op_->DebugString();
    }

    PADDLE_ENFORCE(i0 != nullptr && i1 != nullptr,
                   "Convert to VarHandle error");

    if (level_ == r.level_) {
      return i0->name_ < i1->name_;
    }

    return false;
  }
};
std::unique_ptr<ir::Graph> AllReduceDepsPass::ApplyImpl2(
    std::unique_ptr<ir::Graph> graph) const {
  auto& graph_ops = graph->Get<GraphOps>(kGraphOps);
  OpGraphView graph_view(graph_ops);
  std::unordered_set<OpHandleBase*> cur_level_ops;
  std::vector<std::unordered_set<AllReduceOpHandle*>> allreduce_ops;
  std::unordered_map<AllReduceOpHandle*, int> visited;

  for (auto* op : graph_view.AllOps()) {
    if (graph_view.PrecedingOps(op).empty()) {
      cur_level_ops.insert(op);
    }
  }
  VLOG(11) << "cur_level_ops:" << cur_level_ops.size() << std::endl;

  int level = 0;
  while (!cur_level_ops.empty()) {
    std::unordered_set<OpHandleBase*> next_level_ops;
    std::unordered_set<AllReduceOpHandle*> next_level_allreduce_ops;

    for (auto* op : cur_level_ops) {
      if (op->Name() == "allreduce" || op->Name() == "all_reduce") {
        auto* allreduce_op = dynamic_cast<AllReduceOpHandle*>(op);
        PADDLE_ENFORCE(allreduce_op != nullptr,
                       "Convert to allreduce_op error");
        visited[allreduce_op] = level;
      }

      for (auto* pending_op : graph_view.PendingOps(op)) {
        next_level_ops.insert(pending_op);
      }
    }

    level++;
    cur_level_ops.swap(next_level_ops);
    VLOG(11) << "cur_level_ops:" << cur_level_ops.size() << std::endl;
  }

  std::vector<SortNode> op_list;
  for (auto& it : visited) {
    VLOG(11) << "allreduce_op:" << it.first << ", level:" << it.second;

    SortNode n(it.first, it.second);
    // n.op = it.first;
    // n.level = it.second;
    op_list.push_back(n);
  }

  std::sort(op_list.begin(), op_list.end());

  // add dependency.
  auto& sorted_ops = op_list;
  for (size_t i = 1; i < sorted_ops.size(); ++i) {
    auto* dep_var = new dummyvarhandle(graph->createcontroldepvar());

    auto* pre_op = sorted_ops[i - 1].op_;
    auto* op = sorted_ops[i].op_;

    pre_op->addoutput(dep_var);
    op->addinput(dep_var);
    graph->get<graphdepvars>(kgraphdepvars).emplace(dep_var);

    vlog(10) << "add all_reduce sequential dependencies between " << pre_op
             << " and " << op;
  }

  // VLOG(11) << "allreduce_ops size:" << allreduce_ops.size() << std::endl;

  return graph;
}
*/

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass,
              paddle::framework::details::AllReduceDepsPass)
    .RequirePassAttr(paddle::framework::details::kAllOpDescs);
