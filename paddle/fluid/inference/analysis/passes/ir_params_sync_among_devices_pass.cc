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

#include "paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST) {
    return true;
  }
  return false;
}
}  // namespace
namespace inference {
namespace analysis {

void IrParamsSyncAmongDevicesPass::RunImpl(Argument *argument) {
  PADDLE_ENFORCE(argument->scope_valid());
  PADDLE_ENFORCE(argument->use_gpu_valid());

  platform::Place place;

  // The parameters are on the cpu, therefore, synchronization is not necessary.
  if (!argument->use_gpu()) return;

  LOG(INFO) << "Sync params from CPU to GPU";

  PADDLE_ENFORCE(argument->gpu_device_id_valid());
  place = platform::CUDAPlace(argument->gpu_device_id());

  auto *scope = argument->scope_ptr();
  // Get the program which has been processed by several passes.
  analysis_program_.reset(
      new framework::ProgramDesc(argument->ir_analyzed_program()));

  const auto &global_block = analysis_program_->Block(0);

  // sync the params from cpu to gpu.
  for (auto &var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      std::string var_name = var->Name();
      LOG(INFO) << var_name;
      auto &t = inference::analysis::GetFromScope<framework::LoDTensor>(
          *scope, var_name);

      platform::CPUPlace cpu_place;
      framework::LoDTensor temp_tensor;
      temp_tensor.Resize(t.dims());
      temp_tensor.mutable_data<float>(cpu_place);

      // Copy the parameter data to a tmp tensor.
      TensorCopySync(t, cpu_place, &temp_tensor);
      // Reallocation the space on GPU
      t.mutable_data<float>(place);

      // Copy parameter data to newly allocated GPU space.
      TensorCopySync(temp_tensor, place, &t);
    }
  }
}

std::string IrParamsSyncAmongDevicesPass::repr() const {
  return "ir-params-sync-among-devices-pass";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
