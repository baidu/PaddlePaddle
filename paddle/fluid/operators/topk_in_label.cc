// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/topk_in_label.h"

namespace paddle {
namespace operators {

class TopkinLabelOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Condition"), "Input", "Condition",
                   "TopkinLabel");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "TopkinLabel");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "TopkinLabel");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "TopkinLabel");

    auto cond_dims = ctx->GetInputDim("Condition");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(
        cond_dims, x_dims,
        platform::errors::InvalidArgument(
            "The dims of Inputs(Condition) and Inputs(X) should be same. "
            "But received Condition's shape is [%s], X's shape is [%s]",
            cond_dims, x_dims));
    PADDLE_ENFORCE_EQ(x_dims, y_dims,
                      platform::errors::InvalidArgument(
                          "The dims of Inputs(X) and Inputs(Y) should be same. "
                          "But received X's shape is [%s], Y's shape is [%s]",
                          x_dims, y_dims));

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class TopkinLabelGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Condition"), "Input", "Condition",
                   "TopkinLabel");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "TopkinLabel");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "TopkinLabel");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "TopkinLabel");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

class TopkinLabelOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Condition",
             "(Tensor) A bool tensor whose rank is at least 1. When Condition "
             "is True, yield x, otherwise yield y");
    AddInput("X",
             "(Tensor), The first input tensor of where op. When the "
             "corresponding position of the condition is true, the output "
             "takes the element of X.");
    AddInput("Y",
             "(Tensor), The second input tensor of where op. When the "
             "corresponding position of condition is false, the output takes "
             "the element of Y.");
    AddOutput("Out", "(Tensor), The output tensor of where op.");
    AddComment(R"DOC(
      Where Operator.
      Return a tensor of elements selected from either $X$ or $Y$, depending on condition.
      The equation is:
      $$
      Out_i =
      \begin{cases}
      \X_i, \quad  \text{if} \ cond_i is True \\
      \Y_i, \quad  \text{if} \ cond_i is False \\
      \end{cases}
      $$
)DOC");
  }
};

template <typename T>
class TopkinLabelOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("topk_in_label_grad");
    grad->SetInput("Condition", this->Input("Condition"));
    grad->SetInput("X", this->Input("X"));
    grad->SetInput("Y", this->Input("Y"));
    grad->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(TopkinLabelGradNoNeedBufferVarsInferer, "X",
                                    "Y");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(topk_in_label, ops::TopkinLabelOp, ops::TopkinLabelOpMaker,
                  ops::TopkinLabelOpGradMaker<paddle::framework::OpDesc>,
                  ops::TopkinLabelOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(topk_in_label_grad, ops::TopkinLabelGradOp,
                  ops::TopkinLabelGradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(
    topk_in_label,
    ops::TopkinLabelKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TopkinLabelKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TopkinLabelKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TopkinLabelKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    topk_in_label_grad,
    ops::TopkinLabelGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TopkinLabelGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TopkinLabelGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TopkinLabelGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
