/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <type_traits>
#include "paddle/fluid/operators/jit/kernel_base.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace intrinsic {

void LayerNorm(float* x, float* out, float* mean, float* var,
               const float* scale, const float* bias, int height,
               const float epsilon, int right);

class LayerNormKernel : public KernelMore<LayerNormTuple<float>> {
 public:
  LayerNormKernel() { this->func = LayerNorm; }
  bool UseMe(const typename LayerNormTuple<float>::attr_type&) const override;
  const char* ImplType() const override { return "Intrinsic"; }
};

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle
