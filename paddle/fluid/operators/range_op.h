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
#include <functional>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  PADDLE_ENFORCE(!std::equal_to<T>()(step, 0),
                 "The step of range op shold not be 0.");
  PADDLE_ENFORCE(((start < end) && (step > 0)) || ((start > end) && (step < 0)),
                 "The step shold be greater than 0 while start < end. And the "
                 "step shold be less than 0 while start > end.");
  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

template <typename T>
class CPURangeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    T start = context.Input<framework::Tensor>("Start")->data<T>()[0];
    T end = context.Input<framework::Tensor>("End")->data<T>()[0];
    T step = context.Input<framework::Tensor>("Step")->data<T>()[0];
    auto* out = context.Output<framework::Tensor>("Out");
    int64_t size = 0;
    GetSize(start, end, step, &size);
    out->Resize(framework::make_ddim({size}));
    T* out_data = out->mutable_data<T>(context.GetPlace());
    T value = start;
    for (int64_t i = 0; i < size; ++i) {
      out_data[i] = value;
      value += step;
    }
  }
};

}  // namespace operators
}  // namespace paddle
