// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/* censed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/adagrad_op.h"
#include <vector>

#include <cmath>

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class AdagradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment"),
                   "Input(Moment) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of AdagradOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdagradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(MomentOut) of AdagradOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      "Maybe the Input variable LearningRate has not "
                      "been initialized. You may need to confirm "
                      "if you put exe.run(startup_program) "
                      "after optimizer.minimize function.");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "LearningRate should have one element");
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        "Param and Grad input of AdagradOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment"),
        "Param and Moment input of AdagradOp should have the same dimension.");

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("MomentOut", param_dims);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class AdagradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("Moment", "(Tensor) Second moment");
    AddInput("LearningRate", "(Tensor) Learning rate");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output second moment");

    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-6f);
    AddComment(R"DOC(

Adaptive Gradient Algorithm (Adagrad).

The update is done as follows:

$$moment\_out = moment + grad * grad \\
param\_out = param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}
$$

The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
does not have the epsilon attribute. It is added here in our implementation
as also proposed here: http://cs231n.github.io/neural-networks-3/#ada
for numerical stability to avoid the division by zero error.

)DOC");
  }
};

namespace {
size_t FindPos(const std::vector<int64_t>& rows, int64_t value) {
  return std::find(rows.begin(), rows.end(), value) - rows.begin();
}
}  // namespace

template <typename T>
struct SparseAdagradFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_width = grad.value().dims()[1];
    math::scatter::MergeAdd<platform::CPUDeviceContext, T> merge_func;
    auto grad_merge = merge_func(context, grad);
    auto& merge_rows = grad_merge.rows();
    auto* grad_merge_data = grad_merge.mutable_value()->template data<T>();

    // 2. m += g_m * g_m
    auto grad_square =
        SquareSelectedRows<platform::CPUDeviceContext, T>(context, grad_merge);

    math::SelectedRowsAddToTensor<platform::CPUDeviceContext, T> functor;
    functor(context, grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    for (size_t i = 0; i < merge_rows.size(); i++) {
      for (int64_t j = 0; j < grad_width; j++) {
        param_data[merge_rows[i] * grad_width + j] -=
            lr[0] * grad_merge_data[i * grad_width + j] /
            (std::sqrt(moment_data[merge_rows[i] * grad_width + j]) + epsilon);
      }
    }
  }

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::SelectedRows* moment,
                  framework::SelectedRows* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_width = grad.value().dims()[1];
    math::scatter::MergeAdd<platform::CPUDeviceContext, T> merge_func;
    auto grad_merge = merge_func(context, grad);
    auto& merge_rows = grad_merge.rows();
    auto* grad_merge_data = grad_merge.mutable_value()->template data<T>();

    // 2. m += g_m * g_m
    auto grad_square =
        SquareSelectedRows<platform::CPUDeviceContext, T>(context, grad_merge);

    // math::SelectedRowsAdd<platform::CPUDeviceContext, T> functor;
    // functor(context, grad_square, *moment, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();

    auto* param_data = param->mutable_value()->data<T>();
    auto* moment_data = moment->mutable_value()->data<T>();
    auto* grad_data = grad_square.value().template data<T>();

    for (size_t i = 0; i < merge_rows.size(); i++) {
      for (int64_t j = 0; j < grad_width; j++) {
        auto param_val_idx = param->Index(merge_rows[i]);
        auto grad_val_idx = grad_square.Index(merge_rows[i]);
        auto moment_val_idx = moment->AutoGrownIndex(merge_rows[i], true);

        auto g_m = grad_data[grad_val_idx * grad_width + j] *
                   grad_data[grad_val_idx * grad_width + j];
        moment_data[moment_val_idx * grad_width + j] += g_m;

        param_data[param_val_idx * grad_width + j] -=
            lr[0] * grad_merge_data[i * grad_width + j] /
            (std::sqrt(moment_data[moment_val_idx * grad_width + j]) + epsilon);
      }
    }

    //    for (size_t i = 0; i < merge_rows.size(); i++) {
    //      for (int64_t j = 0; j < grad_width; j++) {
    //        auto param_val_idx = param->Index(merge_rows[i]);
    //        auto moment_val_idx = moment->AutoGrownIndex(merge_rows[i], true);
    //
    //        moment_data[merge_rows[i] * grad_width + j]
    //
    //        param_data[merge_rows[i] * grad_width + j] -=
    //            lr[0] * grad_merge_data[i * grad_width + j] /
    //            (std::sqrt(moment_data[merge_rows[i] * grad_width + j]) +
    //            epsilon);
    //      }
    //    }
  }
};

template struct SparseAdagradFunctor<platform::CPUDeviceContext, float>;
template struct SparseAdagradFunctor<platform::CPUDeviceContext, double>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adagrad, ops::AdagradOp, ops::AdagradOpMaker);
REGISTER_OP_CPU_KERNEL(
    adagrad, ops::AdagradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AdagradOpKernel<paddle::platform::CPUDeviceContext, double>);
