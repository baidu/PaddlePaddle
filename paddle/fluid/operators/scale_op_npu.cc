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
#include <memory>
#include <string>

#include "paddle/fluid/operators/scale_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ScaleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto scale = static_cast<T>(ctx.Attr<float>("scale"));
    auto bias = static_cast<T>(ctx.Attr<float>("bias"));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");
    if (bias_after_scale){
      framework::AttributeMap attr_input= {{"power", 1.0}, {"scale", scale}, {"shift", bias}};
      out->mutable_data<T>(ctx.GetPlace());
      auto runner = NpuOpRunner("Power", {*x}, {*out}, attr_input);

      auto stream =
            ctx.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();
      runner.Run(stream);
    }
    
    // to do else:
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    matmul_v2,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>);
#endif
