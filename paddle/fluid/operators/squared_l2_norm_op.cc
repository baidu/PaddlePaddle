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

#include "paddle/fluid/operators/squared_l2_norm_op.h"

#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;

class SquaredL2NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shold be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) shold be not null.");

    ctx->SetOutputDim("Out", {1});
  }
};

template <typename T>
class SquaredL2NormGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());

    op->SetType("squared_l2_norm_grad");

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    op->SetAttrMap(this->Attrs());
    return op;
  }
};

class SquaredL2NormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shold be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shold be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) shold be not null.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class SquaredL2NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of squared_l2_norm op.");
    AddOutput("Out", "(Scalar) The output of squared_l2_norm op.");
    AddComment(R"DOC(
SquaredL2Norm Operator.

Computes the squared L2 norm of a tensor.

$$Out = \sum_{i} X_{i}^2$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(squared_l2_norm, ops::SquaredL2NormOp,
                  ops::SquaredL2NormOpMaker,
                  ops::SquaredL2NormGradOpMaker<paddle::framework::OpDesc>,
                  ops::SquaredL2NormGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(squared_l2_norm_grad, ops::SquaredL2NormGradOp);
REGISTER_OP_CPU_KERNEL(
    squared_l2_norm,
    ops::SquaredL2NormKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    squared_l2_norm_grad,
    ops::SquaredL2NormGradKernel<paddle::platform::CPUDeviceContext, float>);
