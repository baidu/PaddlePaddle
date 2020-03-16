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

#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
using LoDTensor = framework::LoDTensor;
using LoDTensorBlockingQueueHolder =
    operators::reader::LoDTensorBlockingQueueHolder;

namespace paddle {
namespace operators {

class DequeueOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const std::string& queue_name = Input("blocking_queue");
    auto* queue_holder_var = scope.FindVar(queue_name);
    PADDLE_ENFORCE_NOT_NULL(
        queue_holder_var,
        "No LoDTensorBlockingQueueHolder variable with name %s found",
        queue_name);
    const std::string& var_name = Input("lod_tensor");
    auto* out_tensor = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(in_tensor, "No variable with name %s found",
                            var_name);
    auto* queue_holder =
        queue_holder_var->template GetMutable<LoDTensorBlockingQueueHolder>();

    std::vector<LoDTensor> lod_tensor_vec;
    queue_holder->GetQueue()->Pop(lod_tensor_vec);
    for (int i = 0; i < lod_tensor_vec.size(); ++i) {
      TensorCopySync(lod_tensor_vec[i], dev_place, *out_tensor);
    }
  }
};

class DequeueOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  void Make() override {
    AddInput("blocking_queue",
             "Name of the `LoDTensorBlockingQueueHolder` variable");
    AddInput("lod_tensor", "Name of the `lod_tensor` to assign");
    AddComment(R"DOC(
			Dequeue.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = ::paddle::operators;

REGISTER_FILE_READER_OPERATOR(dequeue, ops::DequeueOp, ops::DequeueOpMaker);
