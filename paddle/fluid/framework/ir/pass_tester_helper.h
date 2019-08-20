/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <sstream>
#include <string>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

struct Layers {
 public:
  const ProgramDesc& main_program() { return program_; }

  VarDesc* data(std::string name) { return lod_tensor(name); }

  VarDesc* mul(VarDesc* x, VarDesc* y) { return binary_op("mul", x, y); }

  VarDesc* elementwise_add(VarDesc* x, VarDesc* y) {
    return binary_op("elementwise_add", x, y);
  }

  VarDesc* dropout(VarDesc* x, float dropout_prob,
                   std::string dropout_implementation) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("dropout");
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("is_test", true);
    op->SetAttr("dropout_prob", dropout_prob);
    op->SetAttr("dropout_implementation", dropout_implementation);
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

 private:
  VarDesc* lod_tensor(std::string name) {
    auto* var = program_.MutableBlock(0)->Var(name);
    var->SetType(proto::VarType::LOD_TENSOR);
    return var;
  }

  VarDesc* binary_op(std::string type, VarDesc* x, VarDesc* y) {
    VarDesc* out = lod_tensor(unique_name());
    OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetInput("Y", {y->Name()});
    op->SetOutput("Out", {out->Name()});
    op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(OpRole::kForward));
    return out;
  }

  std::string unique_name() {
    static int idx = 0;
    return "tmp_" + std::to_string(idx++);
  }

 private:
  ProgramDesc program_;
};

std::string DebugString(Node* node) {
  std::ostringstream os;
  if (node->IsOp() && node->Op()) {
    OpDesc* op = node->Op();
    os << "Op(" << op->Type() << "), inputs:{";
    bool is_first = true;
    for (auto* in : node->inputs) {
      if (!is_first) {
        os << ", ";
      }
      os << in->Name();
      is_first = false;
    }
    os << "}, outputs:{";
    is_first = true;
    for (auto* out : node->outputs) {
      if (!is_first) {
        os << ", ";
      }
      os << out->Name();
      is_first = false;
    }
    os << "}.";
  }
  return os.str();
}

std::string DebugString(const std::unique_ptr<Graph>& graph) {
  std::ostringstream os;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()) {
      os << DebugString(node) << "\n";
    }
  }
  return os.str();
}

int GetNumOpNodes(const std::unique_ptr<Graph>& graph, std::string op_type) {
  int num_nodes = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op() && node->Op()->Type() == op_type) {
      num_nodes++;
    }
  }
  return num_nodes;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
