/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/full_like_op.h"

#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class FullLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FullLikeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FullLikeOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class FullLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of fill-zeros-like op.");
    AddOutput("Out", "The variable will be filled up with specified value.");
    AddAttr<float>("value", "The filled value").SetDefault(0.0);
    AddComment(R"DOC(
FullLike Operator.

Fill up a variable with Attr(value).
The output will have the same shape and dtype as the input.

)DOC");
  }
};

class FullLikeOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};
DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(FullLikeNoNeedBufferVarsInference, "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(
    full_like, ops::FullLikeOp,
    ops::FullLikeOpMaker ops::FullLikeNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(
    full_like, ops::FullLikeKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::FullLikeKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FullLikeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FullLikeKernel<paddle::platform::CPUDeviceContext, bool>);
