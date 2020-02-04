/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Sampler = math::Sampler;
using DDim = framework::DDim;
using LoD = framework::LoD;
using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;

template <typename DeviceContext, typename T>
class TDMSamplerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input_var = context.InputVar("Input");
    auto *travel_var = context.InputVar("Travel");
    auto *layer_var = context.InputVar("Layer");

    auto neg_samples_num_vec =
        context.Attr<std::vector<int>>("neg_samples_num_list");
    auto layer_offset_lod = context.Attr<std::vector<int>>("layer_offset_lod");
    auto output_positive_flag = context.Attr<bool>("output_positive");

    // get all tensor
    auto &input_tensor = input_var->Get<framework::LoDTensor>();
    auto &travel_lod_tensor = travel_var->Get<framework::LoDTensor>();
    auto &layer_lod_tensor = layer_var->Get<framework::LoDTensor>();
    auto *out_var = context.OutputVar("Out");
    auto *label_var = context.OutputVar("Labels");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    auto *label_tensor = label_var->GetMutable<framework::LoDTensor>();
    // get dimension
    int input_ids_num = input_tensor.numel();
    VLOG(1) << "input_ids_num: " << input_ids_num;
    auto layer_nums = neg_samples_num_vec.size();
    VLOG(1) << "layer_nums: " << layer_nums;

    int sample_res_length = 0;
    for (int layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
      sample_res_length += (neg_samples_num_vec[layer_idx] +
                            static_cast<int>(output_positive_flag));
    }
    VLOG(1) << "sample_res_length: " << sample_res_length;

    // get all data
    int *input_data = const_cast<int *>(input_tensor.data<int64_t>());
    int *travel_data = const_cast<int *>(travel_lod_tensor.data<int>());
    int *layer_data = const_cast<int *>(layer_lod_tensor.data<int>());
    auto *output_data = out_tensor->mutable_data<int64_t>(context.GetPlace());
    auto *label_data = label_tensor->mutable_data<int64_t>(context.GetPlace());
    VLOG(1) << "End get input & output data";
    // generate uniform sampler

    auto seed = context.Attr<int>("seed");
    VLOG(1) << "input_ids_num: " << input_ids_num;
    for (int i = 0; i < input_ids_num; ++i) {
      // find leaf node travel path
      auto input_id = input_data[i];
      VLOG(1) << "input_id: " << input_id;
      auto start_offset = input_id * layer_nums;
      VLOG(1) << "Start offset: " << start_offset;
      // nce sample, layer by layer
      int offset = 0;
      for (int layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
        int sample_num = neg_samples_num_vec[layer_idx];
        VLOG(1) << "Sample num: " << sample_num;
        int node_nums =
            layer_offset_lod[layer_idx + 1] - layer_offset_lod[layer_idx];
        VLOG(1) << "node_nums: " << node_nums;
        Sampler *sampler = new math::UniformSampler(node_nums - 1, seed);
        // If output positive, add itself
        if (output_positive_flag) {
          output_data[i * sample_res_length + offset] =
              travel_data[start_offset + layer_idx];
          label_data[i * sample_res_length + offset] =
              travel_data[start_offset + layer_idx] == 0 ? 0 : 1;
          offset += 1;
        }

        // Sampling at layer, until samples enough
        for (int sample_index = 0; sample_index < sample_num; ++sample_index) {
          // Avoid sampling to positive samples
          int64_t sample_res = 0;
          do {
            sample_res = sampler->Sample();
          } while (travel_data[start_offset + layer_idx] ==
                   layer_data[layer_offset_lod[layer_idx] + sample_res]);

          output_data[i * sample_res_length + offset] =
              layer_data[layer_offset_lod[layer_idx] + sample_res];
          VLOG(1) << "sample res: " << sample_res
                  << " layer_offset: " << layer_offset_lod[layer_idx]
                  << " layer res: "
                  << layer_data[layer_offset_lod[layer_idx] + sample_res]
                  << " output res: "
                  << output_data[i * sample_res_length + offset];
          label_data[i * sample_res_length + offset] = 0;
          offset += 1;
        }  // end layer nce
        delete sampler;
      }  // end one input nce
    }    // end all input nce
  }
};

}  // namespace operators
}  // namespace paddle
