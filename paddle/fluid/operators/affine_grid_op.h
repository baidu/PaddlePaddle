/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using Array3 = Eigen::DSizes<int64_t, 3>;
using Array4 = Eigen::DSizes<int64_t, 4>;

/**
 *Return a tensor with evenly spaced numbers over a specified interval.
 */
template <typename DeviceContext, typename T>
struct Linspace {
  framework::Tensor operator()(T start, T end, int count,
                               const framework::ExecutionContext& ctx);
};

template <typename DeviceContext, typename T>
class AffineGridOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* theta = ctx.Input<Tensor>("Theta");
    int n = theta->dims()[0];

    auto size_attr = ctx.Attr<std::vector<int>>("size");
    int h = 0;
    int w = 0;
    if (size_attr.size() == 0) {
      auto* size = ctx.Input<Tensor>("Size");
      Tensor h_sizes;
      framework::TensorCopy(*size, platform::CPUPlace(), &h_sizes);
      const int* h_size_data = h_sizes.data<int>();
      h = h_size_data[2];
      w = h_size_data[3];
    } else {
      h = size_attr[2];
      w = size_attr[3];
    }

    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>({n, h, w, 2}, ctx.GetPlace());

    Linspace<DeviceContext, T> linspace;
    // Get indexes of height with shape [height, width, 1]
    auto h_idx = linspace((T)-1, (T)1, h, ctx);
    auto h_idx_t = EigenTensor<T, 1>::From(h_idx)
                       .reshape(Array2(1, h))
                       .broadcast(Array2(w, 1))
                       .shuffle(Array2(1, 0))
                       .reshape(Array3(h, w, 1));
    LOG(ERROR) << "h_idx_t: " << h_idx_t;
    // Get indexes of width with shape [height, width, 1]
    auto w_idx = linspace((T)-1, (T)1, w, ctx);
    auto w_idx_t = EigenTensor<T, 1>::From(w_idx)
                       .reshape(Array2(1, w))
                       .broadcast(Array2(h, 1))
                       .reshape(Array3(h, w, 1));
    LOG(ERROR) << "w_idx_t: " << w_idx_t;
    // Get constant ones tensor with shape [height, width, 1]
    Tensor ones;
    ones.mutable_data<T>({h, w, 1}, ctx.GetPlace());
    auto ones_t = EigenTensor<T, 3>::From(ones).setConstant((T)1);
    // Get grid tensor with shape [n, h, w, 3] by concatenating h_idx, w_idx and
    // ones
    Tensor grid;
    grid.mutable_data<T>({n, h, w, 3}, ctx.GetPlace());
    auto grid_t = EigenTensor<T, 4>::From(grid);
    grid_t = w_idx_t.concatenate(h_idx_t, 2)
                 .eval()
                 .concatenate(ones_t, 2)
                 .reshape(Array4(1, h, w, 3))
                 .broadcast(Array4(n, 1, 1, 1));
    LOG(ERROR) << "grid_t: " << grid_t;

    // output = grid * theta.T
    // TODO(wanghaoshuang): Refine batched matrix multiply
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    for (int i = 0; i < n; ++i) {
      Tensor sliced_grid = grid.Slice(i, i + 1).Resize({h * w, 3});
      Tensor sliced_theta = theta->Slice(i, i + 1).Resize({2, 3});
      Tensor sliced_out = output->Slice(i, i + 1).Resize({h * w, 2});
      blas.MatMul(sliced_grid, false, sliced_theta, true, T(1), &sliced_out,
                  T(0));
    }
  }
};

template <typename DeviceContext, typename T>
class AffineGridGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto theta_grad = ctx.Output<Tensor>(framework::GradVarName("Theta"));

    int n = output_grad->dims()[0];
    auto size_attr = ctx.Attr<std::vector<int>>("size");
    int h = 0;
    int w = 0;
    if (size_attr.size() == 0) {
      auto* size = ctx.Input<Tensor>("Size");
      Tensor h_sizes;
      framework::TensorCopy(*size, platform::CPUPlace(), &h_sizes);
      const int* h_size_data = h_sizes.data<int>();
      h = h_size_data[2];
      w = h_size_data[3];
    } else {
      h = size_attr[2];
      w = size_attr[3];
    }

    theta_grad->mutable_data<T>({n, 2, 3}, ctx.GetPlace());

    Linspace<DeviceContext, T> linspace;

    // Get indexes of height with shape [height, width, 1]
    auto h_idx = linspace((T)-1, (T)1, h, ctx);
    auto h_idx_t = EigenTensor<T, 1>::From(h_idx)
                       .reshape(Array2(1, h))
                       .broadcast(Array2(w, 1))
                       .shuffle(Array2(1, 0))
                       .reshape(Array3(h, w, 1));
    // Get indexes of width with shape [height, width, 1]
    auto w_idx = linspace((T)-1, (T)1, w, ctx);
    auto w_idx_t = EigenTensor<T, 1>::From(w_idx)
                       .reshape(Array2(1, w))
                       .broadcast(Array2(h, 1))
                       .reshape(Array3(h, w, 1));
    // Get constant ones tensor with shape [height, width, 1]
    Tensor ones;
    ones.mutable_data<T>({h, w, 1}, ctx.GetPlace());
    auto ones_t = EigenTensor<T, 3>::From(ones).setConstant((T)1);
    // Get grid tensor with shape [n, h, w, 3] by concatenating h_idx, w_idx and
    // ones
    Tensor grid;
    grid.mutable_data<T>({n, h, w, 3}, ctx.GetPlace());
    auto grid_t = EigenTensor<T, 4>::From(grid);
    grid_t = w_idx_t.concatenate(h_idx_t, 2)
                 .eval()
                 .concatenate(ones_t, 2)
                 .reshape(Array4(1, h, w, 3))
                 .broadcast(Array4(n, 1, 1, 1));
    // output = grid * theta.T
    // TODO(wanghaoshuang): Refine batched matrix multiply
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    for (int i = 0; i < n; ++i) {
      Tensor sliced_grid = grid.Slice(i, i + 1).Resize({h * w, 3});
      Tensor sliced_out_grad = output_grad->Slice(i, i + 1).Resize({h * w, 2});
      Tensor sliced_theta_grad = theta_grad->Slice(i, i + 1).Resize({2, 3});
      blas.MatMul(sliced_out_grad, true, sliced_grid, false, T(1),
                  &sliced_theta_grad, T(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
