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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_ASCEND_CL)
#include <float.h>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class TrainerDesc;

uint64_t SectionWorker::batch_id_(0);

void SectionWorker::Initialize(const TrainerDesc &desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(
      new ProgramDesc(desc.section_param().section_config().program_desc()));
  for (auto &op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }

  // if not 1F1B scheduler
  if (schedule_mode_ != 1) return;

  for (auto &op : ops_) {
    if (!op->HasAttr("pipeline_send_var")) continue;

    auto op_type = op->Type();
    PADDLE_ENFORCE_EQ(op_type == "send_v2" || op_type == "partial_send", true,
                      platform::errors::PreconditionNotMet(
                          "The op which have `pipeline_send_var` must be "
                          "send_v2 or partial_send op, but this op is %s",
                          op_type));

    auto var_name = op->Attr<std::string>("pipeline_send_var");
    auto send_input_vars = op->InputVars();
    PADDLE_ENFORCE_EQ(
        var_name, send_input_vars[0],
        platform::errors::NotFound("pipeline_send_var %s is not found in op %s",
                                   var_name, op_type));

    int op_role = op->Attr<int>("op_role");
    int FORWARD = static_cast<int>(OpRole::kForward);
    int BACKWARD = static_cast<int>(OpRole::kBackward);
    PADDLE_ENFORCE_EQ(
        op_role == FORWARD || op_role == BACKWARD, true,
        platform::errors::PreconditionNotMet(
            "%s op's op_role must be forward or backward", op_type));

    bool is_last_stage = (pipeline_stage_ == num_pipeline_stages_ - 1);
    bool is_first_stage = (pipeline_stage_ == 0);
    if (op_role == FORWARD && !is_last_stage) {
      // The last pipeline stage does not need to send forward var
      forward_send_vars_.push_back(var_name);
      skip_vars_.push_back(var_name);
    } else if (op_role == BACKWARD && !is_first_stage) {
      // The first pipeline stage does not need to send backward var
      backward_send_vars_.push_back(var_name);
      skip_vars_.push_back(var_name);
    }
  }
}

void SectionWorker::PrepareUnusedVar() {
  VLOG(5) << "begin prepare the unsed vars";
  unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
}

void SectionWorker::RunForward(
    int micro_id, std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    // We run op with op_role = kLRSched only for the first microbatch
    // to avoid increasing the @LR_DECAY_STEP@ multiple times.
    bool run_first_mbatch = (op_role == static_cast<int>(OpRole::kForward)) ||
                            (op_role == (static_cast<int>(OpRole::kForward) |
                                         static_cast<int>(OpRole::kLoss))) ||
                            (op_role == static_cast<int>(OpRole::kLRSched));
    bool run_others = (op_role == static_cast<int>(OpRole::kForward)) ||
                      (op_role == (static_cast<int>(OpRole::kForward) |
                                   static_cast<int>(OpRole::kLoss)));
    if ((micro_id == 0 && run_first_mbatch) || (micro_id != 0 && run_others)) {
      VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*microbatch_scopes_[micro_id], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[micro_id], op.get(),
                            unused_vars_, gc.get());
      }
    }
  }
}

void SectionWorker::RunBackward(
    int micro_id, std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    if ((op_role == static_cast<int>(OpRole::kBackward)) ||
        (op_role == (static_cast<int>(OpRole::kBackward) |
                     static_cast<int>(OpRole::kLoss)))) {
      VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*microbatch_scopes_[micro_id], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[micro_id], op.get(),
                            unused_vars_, gc.get());
      }
    }
  }
}

void SectionWorker::RunUpdate(
    std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "Update: running op " << op->Type();
      op->Run(*microbatch_scopes_[num_microbatches_ - 1], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                            op.get(), unused_vars_, gc.get());
      }
    }
  }
}

void SectionWorker::RunFThenB(std::unique_ptr<GarbageCollector> &gc) {
  // F-then-B scheduler which runs Forward phase for all microbatches,
  // then runs Backward phase for all microbatches.
  // step1: run forward
  for (int i = 0; i < num_microbatches_; ++i) {
    RunForward(i, gc, unused_vars_);
  }
  // step2: run backward
  for (int i = 0; i < num_microbatches_; ++i) {
    RunBackward(i, gc, unused_vars_);
  }
  // step3: run update
  RunUpdate(gc, unused_vars_);
}

void SectionWorker::Run1F1B(std::unique_ptr<GarbageCollector> &gc) {
  // 1F1B scheduler, which runs forward phase and backward phase altertively
  // after startup phase. For a stage, the number of microbatches for
  // startup is num_pipeline_stages_ - pipeline_stage_ - 1, where
  // num_pipeline_stages_ is the total number of pipeline stages and
  // pipeline_stage_ is the pipeline stage of the current device.
  auto startup_steps = num_pipeline_stages_ - pipeline_stage_ - 1;
  VLOG(3) << "startup_steps:" << startup_steps
          << ", num_stages: " << num_pipeline_stages_
          << ", stage:" << pipeline_stage_;
  PADDLE_ENFORCE_GT(
      num_microbatches_, startup_steps,
      platform::errors::InvalidArgument(
          "To use pipeline with 1F1B scheduler, please make sure number of "
          "microbatches (%d) is than startup steps (%d).",
          num_microbatches_, startup_steps));
  int fw_step = 0;
  int bw_step = 0;

  bool is_last_stage = (pipeline_stage_ == num_pipeline_stages_ - 1);
  bool is_first_stage = (pipeline_stage_ == 0);

  int reserve_fw_send_step = 0;
  // startup phase
  while (fw_step < startup_steps) {
    RunForward(fw_step, gc, unused_vars_);
    fw_step += 1;
  }

  // 1f1b phase
  while (fw_step < num_microbatches_) {
    RunForward(fw_step, gc, unused_vars_);

    // delete backward send var at step=(bw_step - 2)
    if (gc && !is_first_stage && bw_step >= 2) {
      DeleteUnusedTensors(*microbatch_scopes_[bw_step - 2], backward_send_vars_,
                          gc.get());
    }

    RunBackward(bw_step, gc, unused_vars_);

    // delete forward send var at step<=(fw_step - 1)
    if (gc && !is_last_stage) {
      for (int i = reserve_fw_send_step; i < fw_step; ++i) {
        DeleteUnusedTensors(*microbatch_scopes_[i], forward_send_vars_,
                            gc.get());
      }
      // because assert(startup_steps < num_microbatches_), so will always
      // run 1F1B, reserve_fw_send_step will be update
      reserve_fw_send_step = fw_step;
    }

    fw_step += 1;
    bw_step += 1;
  }

  int reserve_bw_send_step = bw_step - 2;
  // backward phase
  while (bw_step < num_microbatches_) {
    RunBackward(bw_step, gc, unused_vars_);
    bw_step += 1;

    // NOTE(wangxi): will only execute once
    // delete forward send var at step=(num_microbatches_ - 1)
    if (reserve_fw_send_step < num_microbatches_ && !is_last_stage) {
      DeleteUnusedTensors(*microbatch_scopes_[reserve_fw_send_step],
                          forward_send_vars_, gc.get());
      ++reserve_fw_send_step;
    }
  }

  RunUpdate(gc, unused_vars_);

  if (gc && !is_first_stage) {
    // NOTE(wangxi): program must add sync backward send comm at update
    // delete backward send var
    for (int i = reserve_bw_send_step; i < num_microbatches_; ++i) {
      DeleteUnusedTensors(*microbatch_scopes_[i], backward_send_vars_,
                          gc.get());
    }
  }
}

void SectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  if (max_memory_size >= 0) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
    }
#elif defined(PADDLE_WITH_ASCEND_CL)
    if (IsFastEagerDeletionModeEnabled()) {
      VLOG(4) << "Use unsafe fast gc for NPU.";
      gc.reset(new NPUUnsafeFastGarbageCollector(
          BOOST_GET_CONST(platform::NPUPlace, place_), max_memory_size));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Please set FLAGS_fast_eager_deletion_mode=true to use "
          "GarbageCollector on NPU."));
      // TODO(zhiqiu): fix bugs and enable NPUDefaultStreamGarbageCollector.
      VLOG(4) << "Use default stream gc for NPU.";
      gc.reset(new NPUDefaultStreamGarbageCollector(
          BOOST_GET_CONST(platform::NPUPlace, place_), max_memory_size));
    }
#endif
  }  // max_memory_size >= 0

  if (schedule_mode_ == 0) {
    RunFThenB(gc);
  } else {
    Run1F1B(gc);
  }

  dev_ctx_->Wait();
  ++batch_id_;
}

}  // namespace framework
}  // namespace paddle
#endif
