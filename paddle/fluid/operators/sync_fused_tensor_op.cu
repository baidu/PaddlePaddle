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

#include "paddle/fluid/operators/sync_fused_tensor_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    sync_fused_tensor,
    ops::SyncFusedTensorOpKernel<paddle::platform::CUDADeviceContext,
                                 plat::float16>,
    ops::SyncFusedTensorOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SyncFusedTensorOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SyncFusedTensorOpKernel<paddle::platform::CUDADeviceContext, double>);
