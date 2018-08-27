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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/concat.h"

namespace paddle {
namespace operators {

namespace detail {
template <typename Container>
inline framework::LoD ConcatLoD(const Container &xs,
                                std::vector<framework::Tensor> *xs_in_order) {
  std::vector<size_t> result;
  result.resize(xs[0].get().lod()[0].size());

  for (size_t i = 1; i < result.size(); ++i) {
    size_t sum = 0;
    for (size_t j = 0; j < xs.size(); ++j) {
      auto &x_lod = xs[j].get().lod()[0];
      const framework::Tensor &tensor = xs[j].get();
      xs_in_order->emplace_back(tensor.Slice(x_lod[i - 1], x_lod[i]));
      sum += x_lod[i];
    }
    result[i] = sum;
  }
  framework::LoD lod;
  lod.emplace_back(result);
  return lod;
}
}  // namespace detail

template <typename DeviceContext, typename T>
class SeqConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto xs = detail::VectorRef(context.MultiInput<framework::LoDTensor>("X"),
                                "Cannot find multiple input X");
    auto &out = detail::Ref(context.Output<framework::LoDTensor>("Out"),
                            "Cannot find output");

    size_t lod_size = 0;
    for (auto &x : xs) {
      if (lod_size == 0) {
        lod_size = x.get().lod()[0].size();
      } else {
        PADDLE_ENFORCE_EQ(
            lod_size, x.get().lod()[0].size(),
            "The number of sequence must be same between each input");
      }
    }
    PADDLE_ENFORCE_NE(lod_size, 0, "Each input must have sequence information");

    std::vector<framework::Tensor> x_in_order;
    out.set_lod(detail::ConcatLoD(xs, &x_in_order));
    out.mutable_data<T>(context.GetPlace());
    math::ConcatFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), x_in_order, 0,
            &out);
  }
};

template <typename DeviceContext, typename T>
class SeqConcatGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto xs = context.MultiInput<framework::LoDTensor>("X");
    auto dxs =
        context.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    PADDLE_ENFORCE_EQ(xs.size(), dxs.size());
    for (size_t i = 0; i < dxs.size(); ++i) {
      if (dxs[i] != nullptr) {
        dxs[i]->set_lod(xs[i]->lod());
        dxs[i]->mutable_data<T>(context.GetPlace());
      }
    }
    std::vector<framework::LoDTensor> sliced_x;
    std::vector<boost::variant<boost::blank, framework::Tensor>> sliced_dx;

    for (size_t i = 1; i < xs[0]->lod()[0].size(); ++i) {
      for (size_t j = 0; j < xs.size(); ++j) {
        const framework::LoDTensor *x = xs[j];
        framework::LoDTensor *dx = dxs[j];
        auto &x_lod = x->lod()[0];
        framework::LoDTensor tmp;
        tmp.ShareDataWith(x->Slice(x_lod[i - 1], x_lod[i]));
        sliced_x.emplace_back(tmp);
        if (dx != nullptr) {
          sliced_dx.emplace_back(dx->Slice(x_lod[i - 1], x_lod[i]));
        } else {
          sliced_dx.emplace_back(boost::blank());
        }
      }
    }

    math::ConcatGradFunctor<DeviceContext, T> functor;
    std::vector<const framework::LoDTensor *> sliced_x_ptr;
    std::vector<framework::Tensor *> sliced_dx_ptr;
    for (auto &x : sliced_x) {
      sliced_x_ptr.emplace_back(&x);
    }

    for (auto &dx : sliced_dx) {
      try {
        sliced_dx_ptr.emplace_back(&boost::get<framework::Tensor>(dx));
      } catch (boost::bad_get &) {
        sliced_dx_ptr.emplace_back(nullptr);
      }
    }
    functor(context.template device_context<DeviceContext>(),
            detail::Ref(
                context.Input<framework::Tensor>(framework::GradVarName("Out")),
                "Sequence Concat OG must be set"),
            sliced_x_ptr, 0, &sliced_dx_ptr);
  }
};

}  // namespace operators
}  // namespace paddle
