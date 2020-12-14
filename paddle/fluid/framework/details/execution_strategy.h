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
#include <cstddef>  // for size_t

namespace paddle {
namespace framework {
namespace details {

struct ExecutionStrategy {
  enum ExecutorType { kDefault = 0, kExperimental = 1 };
  enum UseDevice {
    kCPU = 0,
    kCUDA = 1,
    kXPU = 2,
  };

  // num_threads indicates the size of thread pool.
  size_t num_threads_{0};
  // TODO(liuyuhui): There is no need to retain use_cuda_,
  // because of the addition of use_device_.
  // But there are too many places in the codes now use 'use_cuda_',
  // it will be temporarily used and remove it later.
  bool use_cuda_{true};
  UseDevice use_device_{kCUDA};
  // Note that allow_op_delay is invalid now.
  bool allow_op_delay_{false};
  // num_iteration_per_drop_scope indicates how many
  // iterations the framework cleans up a local execution scope.
  // In some models, the value of this parameter has a great
  // influence on the performance(about 15%) of the program.
  size_t num_iteration_per_drop_scope_{100};
  // At present, the kExperimental executor is the fastest in most models.
  ExecutorType type_{kExperimental};
  // This debug option.
  bool dry_run_{false};
  bool thread_barrier_{false};

  // only use with async_ssa_graph_executor
  // and pyreader with data queue
  size_t num_iteration_per_run_{1};
};

}  //  namespace details
}  //  namespace framework
}  //  namespace paddle
