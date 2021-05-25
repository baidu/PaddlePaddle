// Copyright (c) 2021 PaddlePaddle Authors. Any Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_any_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cuh"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

template <typename T>
class BoolReduceAnyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");

    std::vector<int> reduce_dims;
    if (reduce_all) {
      reduce_dims.resize(input->dims().size());
      for (int i = 0; i < reduce_dims.size(); ++i) {
        reduce_dims[i] = i;
      }

    } else {
      for (auto e : dims) {
        reduce_dims.push_back(e >= 0 ? e : e + input->dims().size());
      }
    }

    auto stream = context.cuda_device_context().stream();
    TensorReduceFunc<T, T, CustomLogicalOr<T>, detail::IdentityFunctor<T>>(
        *input, output, reduce_dims, static_cast<T>(false),
        CustomLogicalOr<T>(), detail::IdentityFunctor<T>(), stream);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(reduce_any, ops::BoolReduceAnyKernel<bool>);
