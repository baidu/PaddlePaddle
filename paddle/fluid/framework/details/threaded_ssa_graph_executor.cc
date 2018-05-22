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

#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace framework {
namespace details {
ThreadedSSAGraphExecutor::ThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    std::unique_ptr<SSAGraph> &&graph)
    : SSAGraphExecutor(std::move(graph)),
      pool_(strategy.num_threads_ >= 2 ? new ::ThreadPool(strategy.num_threads_)
                                       : nullptr),
      local_scopes_(local_scopes),
      places_(places),
      fetch_ctxs_(places),
      running_ops_(0),
      strategy_(strategy),
      thread_cnt_(strategy.num_threads_) {}

void ThreadedSSAGraphExecutor::RunOp(
    std::atomic<int> *total_ops, BlockingQueue<OpHandleBase *> *ready_ops,
    std::unordered_map<OpHandleBase *, std::atomic<size_t>> *pending_op_deps) {
  bool timeout;
  OpHandleBase *current_op = nullptr;

  while (true) {
    // 1. If current_op is nullptr, get a runnable op from ready_ops.
    if (current_op == nullptr) {
      if ((*total_ops) <= 0) break;
      current_op = ready_ops->Pop(1, &timeout);
      if (timeout) continue;
    }

    // 2. Run the current op.
    try {
      VLOG(10) << current_op << " " << current_op->Name() << " : "
               << current_op->DebugString();
      current_op->Run(strategy_.use_event_);
      --(*total_ops);
      VLOG(10) << current_op << " " << current_op->Name() << " Done ";
    } catch (platform::EnforceNotMet ex) {
      exception_.reset(new platform::EnforceNotMet(ex));
    } catch (...) {
      LOG(FATAL) << "Unknown exception catched";
    }
    auto released_vars = current_op->Outputs();

    // 3. Decrease the dependency of pending_op_deps. And find the runnable op.
    current_op = nullptr;
    for (auto ready_var : released_vars) {
      for (auto *op : ready_var->pending_ops_) {
        auto dep_num = --pending_op_deps->at(op);
        if (dep_num == 0) {
          bool push_into_ready_ops =
              current_op != nullptr ||
              (op->IsMultiDeviceTransfer() && strategy_.allow_op_delay_);
          if (push_into_ready_ops) {
            ready_ops->Push(op);
          } else {
            current_op = op;
          }
        }
      }
    }
  }
}

FeedFetchList ThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  // Step 1. Insert FetchOps
  std::vector<std::unique_ptr<FetchOpHandle>> fetch_ops;
  std::unordered_set<std::unique_ptr<VarHandleBase>> fetch_dependencies;
  FeedFetchList fetch_data(fetch_tensors.size());

  InsertFetchOps(fetch_tensors, &fetch_ops, &fetch_dependencies, &fetch_data);

  // Step 2. Collect ready_ops and pending_op_deps
  BlockingQueue<OpHandleBase *> ready_ops;
  std::unordered_map<OpHandleBase *, std::atomic<size_t>> pending_op_deps;

  for (auto &op : graph_->ops_) {
    if (op->Inputs().empty()) {
      ready_ops.Push(op.get());
    } else {
      pending_op_deps[op.get()] = op->NoDupInputSize();
    }
  }
  for (auto &op : fetch_ops) {
    pending_op_deps[op.get()] = op->NoDupInputSize();
  }

  auto insert_ready_ops = [&ready_ops, &pending_op_deps](VarHandleBase *op) {
    if (op->generated_op_ == nullptr) {
      for (auto pending_op : op->pending_ops_) {
        --pending_op_deps[pending_op];
        if (pending_op_deps[pending_op] == 0) {
          ready_ops.Push(pending_op);
        }
      }
    }
  };

  // Insert_ready_ops
  for (auto &var_map : graph_->vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        insert_ready_ops(version_pair.get());
      }
    }
  }

  for (auto &var : graph_->dep_vars_) {
    if (var->generated_op_ == nullptr) {
      insert_ready_ops(var.get());
    }
  }

  // according to total_ops to know whether the loop is over
  std::atomic<int> total_ops(
      static_cast<int>(graph_->ops_.size() + fetch_ops.size()));

  // Step 3. Execution
  std::vector<std::thread> workers;
  workers.resize(thread_cnt_);
  for (size_t i = 0; i < thread_cnt_; ++i) {
    workers[i] = std::thread([&total_ops, &ready_ops, &pending_op_deps, this] {
      RunOp(&total_ops, &ready_ops, &pending_op_deps);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  PADDLE_ENFORCE(total_ops <= 0);

  // Wait FetchOps.
  if (!fetch_ops.empty()) {
    fetch_ops.clear();
  }

  return fetch_data;
}

void ThreadedSSAGraphExecutor::InsertFetchOps(
    const std::vector<std::string> &fetch_tensors,
    std::vector<std::unique_ptr<FetchOpHandle>> *fetch_ops,
    std::unordered_set<std::unique_ptr<VarHandleBase>> *fetch_dependencies,
    FeedFetchList *fetch_data) {
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->vars_) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(it->second.rbegin()->get());
      }
    }
  }

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars.at(var_name);
    auto *op = new FetchOpHandle(fetch_data, i, &local_scopes_);
    fetch_ops->emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    auto *fetch_dummy = new DummyVarHandle();
    op->AddOutput(fetch_dummy);
    fetch_dependencies->emplace(fetch_dummy);
  }
}

void ThreadedSSAGraphExecutor::InsertPendingOp(
    std::unordered_map<OpHandleBase *, size_t> *pending_ops,
    OpHandleBase *op_instance) const {
  pending_ops->insert({op_instance, op_instance->NoDupInputSize()});
}

void ThreadedSSAGraphExecutor::InsertPendingVar(
    std::unordered_set<VarHandleBase *> *pending_vars,
    BlockingQueue<VarHandleBase *> *ready_vars, VarHandleBase *var) const {
  pending_vars->insert(var);
  if (var->generated_op_ == nullptr) {
    ready_vars->Push(var);
  }
}
void ThreadedSSAGraphExecutor::RunOp(
    BlockingQueue<VarHandleBase *> *ready_var_q, details::OpHandleBase *op) {
  auto op_run = [ready_var_q, op, this] {
    try {
      VLOG(10) << op << " " << op->Name() << " : " << op->DebugString();
      op->Run(strategy_.use_event_);
      VLOG(10) << op << " " << op->Name() << " Done ";
      running_ops_--;
      ready_var_q->Extend(op->Outputs());
      VLOG(10) << op << " " << op->Name() << "Signal posted";
    } catch (platform::EnforceNotMet ex) {
      exception_.reset(new platform::EnforceNotMet(ex));
    } catch (...) {
      LOG(FATAL) << "Unknown exception catched";
    }
  };
  if (pool_) {
    pool_->enqueue(op_run);
  } else {
    op_run();
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
