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

#include "paddle/fluid/operators/sampling_id_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class SamplingIdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SamplingIdOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SamplingIdOp should not be null.");
    PADDLE_ENFORCE_LT(ctx->Attrs().Get<float>("min"),
                      ctx->Attrs().Get<float>("max"), "min must less then max");

    auto input_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(input_dims.size() == 2,
                   "Input(X, Filter) should be 2-D tensor.");

    auto dim0 = input_dims[0];
    framework::DDim dims = framework::make_ddim({dim0});
    ctx->SetOutputDim("Out", dims);
    ctx->ShareLoD("X", "Out");
  }
};

class SamplingIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of softmax. "
             "2-D with shape [batch_size, input_feature_dimensions].");
    AddOutput("Out", "SamplingId data tensor.");
    AddComment(R"DOC(
SamplingId Operator.
A layer for sampling id from multinomial distribution from the
 input. Sampling one id for one sample.)DOC");
    AddAttr<float>("min", "Minimum value of random. (float, default 0.0).")
        .SetDefault(0.0f);
    AddAttr<float>("max", "Maximun value of random. (float, default 1.0).")
        .SetDefault(1.0f);
    AddAttr<int>(
        "seed",
        "Random seed used for the random number engine. "
        "0 means use a seed generated by the system."
        "Note that if seed is not 0, this operator will "
        "generate the same random numbers every time. (int, default 0).")
        .SetDefault(0);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sampling_id, ops::SamplingIdOp, ops::SamplingIdOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(sampling_id, paddle::operators::SamplingIdKernel<float>,
                       paddle::operators::SamplingIdKernel<double>);
