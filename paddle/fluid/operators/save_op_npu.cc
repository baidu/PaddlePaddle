/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/operators/save_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    save, ops::SaveOpKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SaveOpKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SaveOpKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SaveOpKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::SaveOpKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::SaveOpKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::SaveOpKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);
#endif
