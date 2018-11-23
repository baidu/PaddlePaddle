//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {
struct CollectiveContext {
  std::vector<std::string> end_points_;
  int rank_id_{0};

  std::string String() const {
    std::stringstream ss;
    ss << "end_points_:";
    for (auto e : end_points_) {
      ss << e << ",";
    }

    ss << "rank_id_:" << rank_id_;

    return ss.str();
  }

  static CollectiveContext *GetInstance() {
    std::call_once(init_flag_,
                   [&]() { context_.reset(new CollectiveContext()); });
    return context_.get();
  }

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<CollectiveContext> context_;
};

struct ReduceOpHandle : public OpHandleBase {
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
// CollectiveContext collective_context_;

#ifdef PADDLE_WITH_CUDA
  const platform::NCCLContextMap *nccl_ctxs_;
  ReduceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places,
                 const platform::NCCLContextMap *nccl_ctxs)
      : OpHandleBase(node),
        local_scopes_(local_scopes),
        places_(places),
        nccl_ctxs_(nccl_ctxs) {
    if (nccl_ctxs_) {
      for (auto &p_ctx : nccl_ctxs_->contexts_) {
        this->SetDeviceContext(platform::CUDAPlace(p_ctx.first),
                               p_ctx.second.ctx_.get());
      }
    }
  }
#else
  ReduceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places)
      : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}
#endif

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return true; };

 protected:
  void RunImpl() override;
  void GatherSelectedRows(
      const std::vector<const SelectedRows *> &src_selecte_rows_,
      const std::vector<platform::Place> &in_places,
      const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes,
      VarHandle *out_var_handle, const platform::Place &out_place,
      SelectedRows *dst_selecte_rows);

  void WaitLocalSelectedRows(
      const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes);

  /*
  void GatherRemoteSelectedRows(
      const CollectiveContext& CollectiveContext,
      platform::DeviceContext *dev_ctx, Scope *scope,
      const std::string &var_name,
                                std::vector<const SelectedRows *> *remote);
                                */

  template <typename T>
  std::vector<const T *> GetInputValues(
      const std::vector<VarHandle *> &in_var_handles,
      const std::vector<const Scope *> &var_scopes) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
