/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/op_kernel_type.h"
#include "paddle/framework/tensor.h"
#include "paddle/framework/variable.h"

namespace paddle {
namespace framework {

static std::vector<int> NHWC2NCHW = {0, 3, 1, 2};
static std::vector<int> NCHW2NHWC = {0, 2, 3, 1};

using KernelTypePair = std::pair<OpKernelType, OpKernelType>;

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type,
                     const std::vector<int>& axis, const Tensor& in,
                     Tensor* out);

}  // namespace framework
}  // namespace paddle
