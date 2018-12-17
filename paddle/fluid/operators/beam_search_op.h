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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/beam_search.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class BeamSearchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* ids = context.Input<framework::LoDTensor>("ids");
    auto* scores = context.Input<framework::LoDTensor>("scores");
    auto* pre_ids = context.Input<framework::LoDTensor>("pre_ids");
    auto* pre_scores = context.Input<framework::LoDTensor>("pre_scores");

    PADDLE_ENFORCE_NOT_NULL(ids);
    PADDLE_ENFORCE_NOT_NULL(scores);
    PADDLE_ENFORCE_NOT_NULL(pre_ids);
    PADDLE_ENFORCE_NOT_NULL(pre_scores);

    LOG(INFO) << "<<< pre_ids >>>";
    LOG(INFO) << "  dims: " << pre_ids->dims();
    LOG(INFO) << "  lod:  " << pre_ids->lod();
    LOG(INFO) << "<<< pre_scores >>>";
    LOG(INFO) << "  dims: " << pre_scores->dims();
    LOG(INFO) << "  lod:  " << pre_scores->lod();
    LOG(INFO) << "<<< ids >>>";
    LOG(INFO) << "  dims: " << ids->dims();
    LOG(INFO) << "  lod:  " << ids->lod();
    LOG(INFO) << "<<< scores >>>";
    LOG(INFO) << "  dims: " << scores->dims();
    LOG(INFO) << "  lod:  " << scores->lod();

    size_t level = context.Attr<int>("level");
    size_t beam_size = context.Attr<int>("beam_size");
    int end_id = context.Attr<int>("end_id");

    LOG(INFO) << "level: " << level << ", beam_size: " << beam_size
              << ", end_id:" << end_id;

    auto selected_ids = context.Output<framework::LoDTensor>("selected_ids");
    auto selected_scores =
        context.Output<framework::LoDTensor>("selected_scores");
    PADDLE_ENFORCE_NOT_NULL(selected_ids);
    PADDLE_ENFORCE_NOT_NULL(selected_scores);

    math::BeamSearchFunctor<DeviceContext, T> alg;
    alg(context.template device_context<DeviceContext>(), *pre_ids, *pre_scores,
        *ids, *scores, selected_ids, selected_scores, level, beam_size, end_id);
  }
};

}  // namespace operators
}  // namespace paddle
