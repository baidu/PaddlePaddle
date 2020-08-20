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

#include <thrust/random.h>
#include <thrust/transform.h>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/bernoulli_op.h"

namespace paddle {
namespace operators {
// it can be consistent with cpu when CUDAGenerator is provided.
template <typename T>
struct BernoulliCudaFunctor {
  unsigned int seed_;
  __host__ __device__ BernoulliCudaFunctor(int seed) : seed_(seed) {}

  __host__ __device__ T operator()(T p) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(0.0, 1.0);
    return static_cast<T>(dist(rng) < p);
  }
};

template <typename T>
class BernoulliOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::random_device rd;
    seed = rd();
    const auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto* in_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());

    int64_t size = x->numel();

    thrust::transform(
        thrust::device_ptr<T>(in_data), thrust::device_ptr<T>(in_data) + size,
        thrust::device_ptr<T>(out_data), GaussianGenerator<T>(seed));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CPU_KERNEL(bernoulli,
                       ops::BernoulliOpKernel<plat::CUDADeviceContext, float>,
                       ops::BernoulliOpKernel<plat::CUDADeviceContext, double>);
