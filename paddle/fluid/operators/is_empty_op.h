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

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

constexpr char kInput[] = "X";
constexpr char kOutput[] = "Out";

template <typename DeviceContext, typename T>
class IsEmptyOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get input
    auto* input_tensor = context.Input<framework::LoDTensor>(kInput);
    // get output
    auto* output_tensor = context.Output<framework::LoDTensor>(kOutput);

    output_tensor->Resize({1});
    output_tensor->mutable_data<bool>(platform::CPUPlace())[0] =
        framework::product(input_tensor->dims()) == 0;
  }
};

}  // namespace operators
}  // namespace paddle
