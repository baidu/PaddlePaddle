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

#pragma once

#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/assert.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
template <typename T>
using Vector = framework::CPUVector<T>;

template <typename T>
class FilterInstagKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // X1 is global FC output
    // Dim [batch size, embedding size]
    auto* x1 = context.Input<LoDTensor>("X1");
    // X2 is ins tag list
    // LoD [[0, Sum(ins1), Sum(ins1, ins2), ... ]]
    auto* x2 = context.Input<LoDTensor>("X2");
    // X3 is local fc tag list
    // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
    auto* x3 = context.Input<Tensor>("X3");

    std::unordered_set<int64_t> filter_tag;
    auto* x3_data = x3->data<int64_t>();
    size_t len = x3->dims()[0];
    for (size_t i = 0; i < len; i++) {
      filter_tag.insert(x3_data[i]);
    }

    // expected auto = const int64_t
    auto* x2_data = x2->data<int64_t>();
    // e.g get [0, 1, 2, 3, ...]
    auto x2_lods = x2->lod()[0];
    auto x1_lods = x1->lod()[0];

    std::vector<size_t> ins_after_filter;
    Vector<size_t> out_lods(1, 0);
    for (size_t i = 0; i < x2_lods.size() - 1; i++) {
      for (size_t j = x2_lods[i]; j < x2_lods[i + 1]; j++) {
        if (filter_tag.find(x2_data[j]) != filter_tag.end()) {
          ins_after_filter.push_back(x2_lods[i]);
          size_t batch_len = x1_lods[i + 1] - x1_lods[i];
          out_lods.push_back(out_lods.back() + batch_len);
          break;
        }
      }
    }

    // set output value
    // for those whose ins been dropout, set 0 for whole lines.
    // otherwise, copy whole line
    // Dim [local fc count, batch size, embedding size]
    LoDTensor* out = context.Output<LoDTensor>("Out");
    LoDTensor* map = context.Output<LoDTensor>("Map");
    // expected auto = const T
    auto* x1_data = x1->data<T>();
    // expected auto = T
    size_t x1_embed_size = x1->dims()[1];
    out->Resize(framework::make_ddim(
        {(int64_t)out_lods.back(), (int64_t)x1_embed_size}));
    map->Resize(framework::make_ddim({(int64_t)ins_after_filter.size(), 2}));
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    auto* map_data = map->mutable_data<int64_t>(context.GetPlace());

    Vector<size_t> map_lods;
    for (size_t i = 0; i < ins_after_filter.size(); i++) {
      map_data[i * 2] = i;
      map_data[i * 2 + 1] = (int64_t)ins_after_filter[i];
      map_lods.push_back(i);
    }
    map_lods.push_back(ins_after_filter.size());
    std::vector<Vector<size_t>> map_lod_info;
    map_lod_info.push_back(map_lods);

    map->set_lod(map_lod_info);

    std::vector<Vector<size_t>> out_lod_info;
    out_lod_info.push_back(out_lods);
    out->set_lod(out_lod_info);
    memset(out_data, 0, out->dims()[0] * out->dims()[1] * sizeof(T));
    for (size_t i = 0; i < ins_after_filter.size(); i++) {
      size_t pos = out_lods[i];
      for (size_t k = x1_lods[ins_after_filter[i]];
           k < x1_lods[ins_after_filter[i] + 1]; k++) {
        memcpy(out_data + pos * x1_embed_size, x1_data + k * x1_embed_size,
               x1_embed_size * sizeof(T));
        ++pos;
      }
    }
  }
};

template <typename T>
class FilterInstagGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x1_grad = context.Output<LoDTensor>(framework::GradVarName("X1"));
    auto* x1 = context.Input<LoDTensor>("X1");
    auto* mmap = context.Input<LoDTensor>("Map");
    x1_grad->set_lod(x1->lod());
    x1_grad->Resize(x1->dims());
    auto mmap_data = mmap->data<int64_t>();
    // expected auto = T
    auto* output_grad_data = output_grad->data<T>();
    // expected auto = T
    auto* x1_grad_data = x1_grad->mutable_data<T>(context.GetPlace());
    memset(x1_grad_data, 0, x1->dims()[0] * x1->dims()[1] * sizeof(T));
    auto output_dims = output_grad->dims();
    for (size_t i = 0; i < output_dims[0]; i++) {
      int src_ln = mmap_data[i * 2], dst_ln = mmap_data[i * 2 + 1];
      for (size_t j = 0; j < output_dims[1]; j++) {
        x1_grad_data[dst_ln * output_dims[1] + j] =
            output_grad_data[src_ln * output_dims[1] + j];
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
