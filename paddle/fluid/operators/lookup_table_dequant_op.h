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

#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/math/blas.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/fluid/operators/distributed/parameter_prefetch.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

float *dequant(const unsigned char *in, float *out, float min, float max,
               int emb_size, int pow_2_bits) {
  float scale = (max - min) / pow_2_bits;
  for (int i = 0; i < emb_size; ++i) {
    float x = scale * static_cast<int>(in[i]) + min;
    out[i] = x;
  }
  return out;
}

constexpr int64_t kNoPadding = -1;

template <typename T>
class LookupTableDequantKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ids_t = context.Input<LoDTensor>("Ids");      // int tensor
    auto *output_t = context.Output<LoDTensor>("Out");  // float tensor
    auto *table_var = context.InputVar("W");

    auto id_name = context.InputNames("Ids").front();
    auto embedding_name = context.InputNames("W").front();
    auto out_name = context.OutputNames("Out").front();

    int64_t padding_idx = context.Attr<int64_t>("padding_idx");
    int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
    int64_t ids_numel = ids_t->numel();

    if (table_var->IsType<LoDTensor>()) {
      auto *table_t = context.Input<LoDTensor>("W");
      int64_t row_number = table_t->dims()[0];
      int64_t quant_number = table_t->dims()[1];
      int64_t row_width = (quant_number - 2) * 4;

      auto *table = table_t->data<T>();
      auto *output = output_t->mutable_data<T>(context.GetPlace());
      int pow_2_bits = static_cast<int>(pow(2, 8));
      for (int64_t i = 0; i < ids_numel; ++i) {
        if (padding_idx != kNoPadding && ids[i] == padding_idx) {
          memset(output + i * row_width, 0, row_width * sizeof(T));
        } else {
          PADDLE_ENFORCE_LT(
              ids[i], row_number,
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              row_number, ids[i]);
          PADDLE_ENFORCE_GE(
              ids[i], 0,
              "Variable value (input) of OP(fluid.layers.embedding) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              row_number, ids[i]);
          float min = *(table + ids[i] * quant_number);
          float max = *(table + ids[i] * quant_number + 1);
          int offset = ids[i] * quant_number + 2;
          const unsigned char *tensor_buf =
              reinterpret_cast<const unsigned char *>(table + offset);
          dequant(tensor_buf, output + i * row_width, min, max, row_width,
                  pow_2_bits);
        }
      }
    } else if (table_var->IsType<SelectedRows>()) {
      // not supportd
      PADDLE_ENFORCE_EQ(1, 2);
    }
  }
};

}  // namespace operators
}  // namespace paddle
