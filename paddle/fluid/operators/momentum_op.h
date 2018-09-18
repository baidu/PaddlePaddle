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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto grad = ctx.Input<framework::Tensor>("Grad");
    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    bool use_lars = ctx.Attr<bool>("use_lars");
    T lars_coeff = ctx.Attr<float>("lars_coeff");
    T lars_weight_decay = ctx.Attr<float>("lars_weight_decay");

    auto p_out = framework::EigenVector<T>::Flatten(*param_out);
    auto v_out = framework::EigenVector<T>::Flatten(*velocity_out);

    auto p = framework::EigenVector<T>::Flatten(*param);
    auto v = framework::EigenVector<T>::Flatten(*velocity);
    auto g = framework::EigenVector<T>::Flatten(*grad);
    auto* lr = learning_rate->data<T>();

    if (!use_lars) {
      v_out = v * mu + g;
      if (use_nesterov) {
        p_out = p - (g + v_out * mu) * lr[0];
      } else {
        p_out = p - lr[0] * v_out;
      }
    } else {
      framework::Tensor p_norm_t, g_norm_t;
      p_norm_t.Resize({1});
      g_norm_t.Resize({1});
      p_norm_t.mutable_data<T>(ctx.GetPlace());
      g_norm_t.mutable_data<T>(ctx.GetPlace());
      auto ep_norm = framework::EigenScalar<T>::From(p_norm_t);
      auto eg_norm = framework::EigenScalar<T>::From(g_norm_t);

      ep_norm = p.square().sum().sqrt();
      eg_norm = g.square().sum().sqrt();
      T local_lr = lr[0];
      if (ep_norm(0) > 0 && eg_norm(0) > 0) {
        local_lr = lr[0] * lars_coeff * ep_norm(0) /
                   (eg_norm(0) + lars_weight_decay * ep_norm(0));
      }
      v_out = v * mu + local_lr * (g + lars_weight_decay * p);
      p_out = p - v_out;
    }
  }
};

}  // namespace operators
}  // namespace paddle
