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

#include "paddle/operators/fill_zeros_like_op.h"
#include "paddle/framework/op_registry.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fill_zeros_like,
    ops::FillZerosLikeKernel<paddle::platform::CUDADeviceContext, int>,
    ops::FillZerosLikeKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::FillZerosLikeKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FillZerosLikeKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FillZerosLikeKernel<paddle::platform::CUDADeviceContext, bool>);
