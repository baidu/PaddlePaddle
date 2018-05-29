/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_FLUID_OPERATORS_LISTEN_AND_SERV_OP_H_
#define PADDLE_FLUID_OPERATORS_LISTEN_AND_SERV_OP_H_

#pragma once

#include <stdint.h>
#include <atomic>
#include <set>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/grpc_server.h"

namespace paddle {
namespace operators {

constexpr char kOptimizeBlock[] = "OptimizeBlock";
constexpr char kPrefetchBlock[] = "PrefetchBlock";

void RunServer(std::shared_ptr<detail::AsyncGRPCServer> service);

class ListenAndServOp : public framework::OperatorBase {
 public:
  ListenAndServOp(const std::string& type,
                  const framework::VariableNameMap& inputs,
                  const framework::VariableNameMap& outputs,
                  const framework::AttributeMap& attrs);

  void RunSyncLoop(framework::Executor* executor,
                   framework::ProgramDesc* program,
                   framework::Scope* recv_scope,
                   framework::BlockDesc* prefetch_block) const;

  void RunAsyncLoop(framework::Executor* executor,
                    framework::ProgramDesc* program) const;

  void SavePort() const;

  void WaitServerReady();

  int GetSelectedPort() { return selected_port_; }

  void Stop() override;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override;

  static void ResetPort() { selected_port_ = 0; }

 protected:
  mutable std::shared_ptr<detail::AsyncGRPCServer> rpc_service_;
  mutable std::shared_ptr<std::thread> server_thread_;
  // FIXME(wuyi): it's static so that the operator can be cloned.
  static std::atomic_int selected_port_;
};

class SignalHandler {
 public:
  typedef std::shared_ptr<detail::ReceivedQueue> BlockingQueue;
  typedef std::unordered_set<BlockingQueue> BlockingQueueSet;

 public:
  // SignalHandler is noncopyable and can NOT be constructed
  SignalHandler() = delete;
  SignalHandler(const SignalHandler&) = delete;
  SignalHandler& operator=(const SignalHandler&) = delete;

  static void StopAndExit(int signal_num);

  static void RegisterBlockingQueue(BlockingQueue&);

  static inline bool IsProgramExit() { return program_exit_flag_; }

 private:
  static bool program_exit_flag_;

  static BlockingQueueSet blocking_queue_set_;
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_FLUID_OPERATORS_LISTEN_AND_SERV_OP_H_
