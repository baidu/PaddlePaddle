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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
struct GetTensorValue {
  T operator()(const framework::Tensor& tensor) const;
};

template <typename DeviceContext, typename T>
class AllcloseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // get attrs
    bool equal_nan = ctx.Attr<bool>("equal_nan");
    // get input/output
    const auto* input = ctx.Input<Tensor>("Input");
    const auto* other = ctx.Input<Tensor>("Other");
    const auto* rtol = ctx.Input<Tensor>("Rtol");
    const auto* atol = ctx.Input<Tensor>("Atol");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());
    // get place
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    auto input_v = framework::EigenVector<T>::Flatten(*input);
    auto other_v = framework::EigenVector<T>::Flatten(*other);
    auto out_v = framework::EigenScalar<bool>::From(*out);

    GetTensorValue<DeviceContext, double> get_tensor_value;
    double rtol_v = get_tensor_value(*rtol);
    double atol_v = get_tensor_value(*atol);

    auto left = (input_v - other_v).abs();
    auto right = atol_v + rtol_v * other_v.abs();
    auto compare_res = left <= right;

    if (equal_nan) {
      auto input_nan = input_v.isnan();
      auto other_nan = other_v.isnan();
      out_v.device(place) =
          (input_nan == other_nan).all() && (compare_res != input_nan).all();
    } else {
      out_v.device(place) = compare_res.all();
    }
  }
};

}  // namespace operators
}  // namespace paddle
