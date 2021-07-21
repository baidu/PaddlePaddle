/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class Fill_OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill an tensor inplace with `value` and `shape`. The type of the tensor is specify by
                `dtype`.
                )DOC");
    AddInput("X", "(Tensor) The input tensor.");
    AddOutput("Out",
              "Tensor, the clipped tensor, with the same shape and data type "
              "as input(x)");
    AddInput(
        "value",
        "The float values of tensor, whose dim is one, and no need of grad");
  }
};

class Fill_Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "Fill_");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "Fill_");
    auto x_dims = context->GetInputDim("X");
    context->SetOutputDim("Out", x_dims);
    // context->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class Fill_OpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

template <typename T>
class Fill_Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *in = ctx.Input<framework::Tensor>("X");
    auto *value = ctx.Input<framework::Tensor>("value");
    auto *out = ctx.Output<framework::Tensor>("Out");
    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    auto fill_val = *(value->data<T>());
    std::fill(out_data, out_data + in->numel(), fill_val);
  }
};

class Fill_GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mul");
    auto x_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

template <typename T>
class Fill_GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType(this->ForwardOpType() + "_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

template <typename T>
class Fill_GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    if (dx) {
      auto *data = dx->mutable_data<T>(ctx.GetPlace());
      std::fill(data, data + dx->numel(), T(0));
    }
  }
};

DECLARE_INPLACE_OP_INFERER(Fill_OpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(Fill_GradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(fill_inplace, ops::Fill_Op, ops::Fill_OpMaker,
                  ops::Fill_OpVarTypeInference,
                  ops::Fill_GradOpMaker<paddle::framework::OpDesc>,
                  ops::Fill_GradOpMaker<paddle::imperative::OpBase>,
                  ops::Fill_OpInplaceInferer);

REGISTER_OPERATOR(fill_inplace_grad, ops::Fill_GradOp,
                  ops::Fill_GradInplaceInferer);

REGISTER_OP_CPU_KERNEL(fill_inplace, ops::Fill_Kernel<float>,
                       ops::Fill_Kernel<double>, ops::Fill_Kernel<int64_t>,
                       ops::Fill_Kernel<int>,
                       ops::Fill_Kernel<paddle::platform::float16>,
                       ops::Fill_Kernel<bool>);

REGISTER_OP_CPU_KERNEL(fill_inplace_grad, ops::Fill_GradKernel<float>,
                       ops::Fill_GradKernel<double>,
                       ops::Fill_GradKernel<int64_t>, ops::Fill_GradKernel<int>,
                       ops::Fill_GradKernel<paddle::platform::float16>,
                       ops::Fill_GradKernel<bool>);
