/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/scatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class GatherNdOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      "This kernel only runs on CPU.");

    auto *x = ctx.Input<Tensor>("X");
    auto *indice = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Out");

    output->mutable_data<T>(ctx.GetPlace());
    if (x->numel() == 0) return;

    const auto &indice_type = indice->type();
    bool indice_type_match = indice_type == framework::proto::VarType::INT32 ||
                            indice_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(
        indice_type_match, true,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(indice_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));
    if (indice_type == framework::proto::VarType::INT32) {
      CPUGatherNd<T, int>(ctx.device_context(), *x, *indice, output);
    } else if (indice_type == framework::proto::VarType::INT64) {
      CPUGatherNd<T, int64_t>(ctx.device_context(), *x, *indice, output);
    }
  }
};

template <typename T>
class GatherNdGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      "This kernel only runs on CPU.");
    auto *indice = ctx.Input<Tensor>("Index");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));
    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;

    const auto &indice_type = indice->type();
    bool indice_type_match = indice_type == framework::proto::VarType::INT32 ||
                            indice_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(
        indice_type_match, true,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(indice_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));
    if (indice_type == framework::proto::VarType::INT32) {
      ScatterNdAdd<T, int32_t>(ctx, *dO, *indice, dX);
    } else if (indice_type == framework::proto::VarType::INT64) {
      ScatterNdAdd<T, int64_t>(ctx, *dO, *indice, dX);
    }
  }
};

}  // namespace operators
}  // namespace paddle
