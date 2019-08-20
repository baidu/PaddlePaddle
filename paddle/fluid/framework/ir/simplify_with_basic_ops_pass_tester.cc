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

#include "paddle/fluid/framework/ir/simplify_with_basic_ops_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(SimplifyWithBasicOpsPass, dropout) {
  Layers layers;
  // (x, y) -> mul -> tmp_0
  // (tmp_0) -> dropout -> (tmp_1)
  // (tmp_1, z) -> elementwise_add -> (tmp_2)
  auto* x = layers.data("x");
  auto* y = layers.data("y");
  auto* z = layers.data("z");
  auto* mul_out = layers.mul(x, y);
  auto* dropout_out = layers.dropout(mul_out, 0.5f, "downgrade_in_infer");
  auto* out = layers.elementwise_add(dropout_out, z);
  LOG(INFO) << "out: " << out->Name();

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("simplify_with_basic_ops_pass");
  int num_nodes_before = graph->Nodes().size();
  int num_dropout_nodes_before = GetNumOpNodes(graph, "dropout");
  int num_scale_nodes_before = GetNumOpNodes(graph, "scale");
  LOG(INFO) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_dropout_nodes_after = GetNumOpNodes(graph, "dropout");
  int num_scale_nodes_after = GetNumOpNodes(graph, "scale");
  LOG(INFO) << DebugString(graph);
  LOG(INFO) << "num_nodes_before: " << num_nodes_before
            << ", num_dropout_nodes_before: " << num_dropout_nodes_before
            << ", num_scale_nodes_before: " << num_scale_nodes_before;
  LOG(INFO) << "num_nodes_after: " << num_nodes_after
            << ", num_dropout_nodes_after: " << num_dropout_nodes_after
            << ", num_scale_nodes_after: " << num_scale_nodes_after;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(simplify_with_basic_ops_pass);
