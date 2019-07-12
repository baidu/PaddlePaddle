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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/prelu.h"
#include "paddle/fluid/operators/prelu_op.h"
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

static const int CUDA_NUM_THREADS = 1024;
static const int CUDA_MAX_NUM_BLOCKS = 65535;

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CUDAPReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* out = context.Output<Tensor>("Out");

    const T* x_ptr = x->data<T>();
    T* o_ptr = out->mutable_data<T>(context.GetPlace());

    const T* alpha_ptr = alpha->data<T>();
    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();
    std::vector<int> input_shape = framework::vectorize2int(dim);

    if (mode == "channel") {
      math::PreluChannelWiseDirectCUDAFunctor<T> prelu_channel_wise;
      prelu_channel_wise(context.cuda_device_context().stream(), x_ptr,
                         alpha_ptr, o_ptr, input_shape);
    } else if (mode == "element") {
      math::PreluElementWiseDirectCUDAFunctor<T> prelu_element_wise;
      prelu_element_wise(context.cuda_device_context().stream(), x_ptr,
                         alpha_ptr, o_ptr, input_shape);
    } else {
      math::PreluScalarDirectCUDAFunctor<T> prelu_scalar;
      prelu_scalar(context.cuda_device_context().stream(), x_ptr, alpha_ptr,
                   o_ptr, input_shape);
    }
  }
};

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

template <typename T>
__global__ void PReluGradElementWiseKernel(const T* x_ptr_, const T* y_ptr_,
                                           const T* alpha_ptr_,
                                           const T* dy_ptr_, T* dx_ptr_,
                                           T* dalpha_ptr_, size_t channel,
                                           size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T* x_ptr = x_ptr_ + offset;
  const T* y_ptr = y_ptr_ + offset;
  const T* dy_ptr = dy_ptr_ + offset;
  const T* alpha_ptr = y_ptr_ + offset;
  T* dx_ptr = dx_ptr_ + offset;
  T* dalpha_ptr = dalpha_ptr_ + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T y = y_ptr[i];
    T x = x_ptr[i];
    T alpha = alpha_ptr[i];
    T dy = dy_ptr[i];
    if (dx_ptr != nullptr) dx_ptr[i] = (y > 0) ? dy : alpha * dy;
    if (dalpha_ptr != nullptr) dalpha_ptr[i] = (x > 0) ? 0 : dy;
  }
}

template <typename T>
class PreluGradElementwiseFunctor {
 public:
  void operator()(cudaStream_t stream, const T* x, const T* y, const T* alpha,
                  const T* dy, T* dx, T* dalpha, std::vector<int> input_shape) {
    size_t unroll = input_shape[0] * input_shape[1];
    size_t spatial_size = input_shape[2] * input_shape[3];
    CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
    PReluGradElementWiseKernel<T><<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
        x, y, alpha, dy, dx, dalpha, input_shape[1], spatial_size);
  }
};

template <typename DeviceContext, typename T>
class CUDAPReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Out");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dalpha = context.Output<Tensor>(framework::GradVarName("Alpha"));

    const T* x_ptr = x->data<T>();
    const T* y_ptr = y->data<T>();
    const T* alpha_ptr = alpha->data<T>();
    const T* dy_ptr = dy->data<T>();
    T* dx_ptr = dx ? dx->mutable_data<T>(context.GetPlace()) : nullptr;
    T* dalpha_ptr =
        dalpha ? dalpha->mutable_data<T>(context.GetPlace()) : nullptr;

    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();
    std::vector<int> input_shape = framework::vectorize2int(dim);

    framework::DDim input_dim_(framework::make_ddim(input_shape));
    auto stream = context.cuda_device_context().stream();

    T* dalpha_tmp_ptr;
    Tensor dalpha_tmp;
    if (mode == "element") {
      dalpha_tmp_ptr = dalpha_ptr;
    } else if (dalpha_ptr == nullptr) {
      dalpha_tmp_ptr = nullptr;
    } else {
      auto& dev_ctx = context.template device_context<DeviceContext>();
      dalpha_tmp =
          context.AllocateTmpTensor<T, DeviceContext>(input_dim_, dev_ctx);
      dalpha_tmp_ptr = dalpha_tmp.mutable_data<T>(context.GetPlace());
    }

    if (mode == "element" || dalpha_ptr == nullptr) {
      PreluGradElementwiseFunctor<T> prelu_element_wise_grad;
      prelu_element_wise_grad(stream, x_ptr, y_ptr, alpha_ptr, dy_ptr, dx_ptr,
                              dalpha_tmp_ptr, input_shape);
      return;
    }

    std::vector<int> reduce_dims;
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (mode == "channel" && i == 1) continue;
      reduce_dims.push_back(i);
    }

    TensorReduce<T, T, cub::Sum, IdentityFunctor<T>>(
        dalpha_tmp, dalpha, reduce_dims, static_cast<T>(0), cub::Sum(),
        IdentityFunctor<T>(), stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    prelu, ops::CUDAPReluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAPReluKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    prelu_grad,
    ops::CUDAPReluGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAPReluGradKernel<paddle::platform::CUDADeviceContext, double>);
