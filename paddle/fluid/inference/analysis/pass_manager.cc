/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/pass_manager.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

void PassManagerMain::RunAll(const framework::proto::ProgramDesc &desc) {
  for (auto &pass : data_) {
    pass->RunAll();
  }
}

//
// CustomIterPassManager
//

DataFlowGraphPassManager::DataFlowGraphPassManager() {
  type_ = kCustomIter;
  Register("fluid_to_data_flow_graph", new FluidToDataFlowGraphPass);
}

void DataFlowGraphPassManager::RunAll() {
  for (auto &pass : data_) {
    pass->Run(graph_);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
