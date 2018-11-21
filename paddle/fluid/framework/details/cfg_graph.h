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

#pragma once
#include <algorithm>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/details/memory_reuse_types.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

class ControlFlowGraph {
 public:
  ControlFlowGraph() {}
  explicit ControlFlowGraph(const ir::Graph& graph);

  void LiveVariableAnalysis();

  void UpdateGraph(const std::string& old_node, const std::string& new_node,
                   int begin_idx);

  const std::unordered_set<std::string>& LiveIn(ir::Node* op) const;
  const std::unordered_set<std::string>& LiveOut(ir::Node* op) const;
  const std::unordered_set<std::string>& Def(ir::Node* op) const;
  const std::unordered_set<std::string>& Use(ir::Node* op) const;
  const std::vector<ir::Node*>& Ops() const;
  std::vector<ir::Node*>& Ops();

  // for ssa-graph nodes
  ir::Node* GetNodeFromVarName(const std::string& name, ir::Node* op) const;

 private:
  void ConnectNodes();
  using NodeListMap = std::unordered_map<ir::Node*, std::list<ir::Node*>>;
  using VarSetMap =
      std::unordered_map<ir::Node*, std::unordered_set<std::string>>;
  // successors ops use the output variables.
  NodeListMap successors_;
  // predecessors ops generated input variables.
  NodeListMap predecessors_;
  // variables lived before run current op.
  VarSetMap live_in_;
  // variables lived after run current op.
  VarSetMap live_out_;
  VarSetMap uses_;              // op inputs
  VarSetMap defs_;              // op outputs
  std::vector<ir::Node*> ops_;  // op sequence by topology sort
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
