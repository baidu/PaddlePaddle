/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
inline void detail(std::string name, T t) {
  // printf("in_c:%s %f %f %f %f %f\n", name.c_str(), t[0], t[1], t[2], t[3],
  //       t[4]);

  printf("in_c:%s %f %f %f %f %f\n", name.c_str(), t(0), t(1), t(2), t(3),
         t(4));
}

template <typename Place, typename T>
class LRNKernel : public framework::OpKernel {
 public:
  using Tensor = framework::Tensor;

  // f(x) = x * ( k + alpha * SUM((x)^2) )^(-beta)
  // x represents inputs
  // f(x) represents outputs
  void Compute(const framework::ExecutionContext& ctx) const override {
    // input
    const Tensor* x = ctx.Input<Tensor>("X");
    auto x_dims = x->dims();

    // NCHW
    int N = x_dims[0];
    int C = x_dims[1];
    int H = x_dims[2];
    int W = x_dims[3];
    int one_img_size = H * W;
    int one_sample_size = C * one_img_size;
    printf("N:%d one_img_size:%d\n", N, one_sample_size);

    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    // mid_out save the intermediate result for backward
    Tensor* mid_out = ctx.Output<Tensor>("mid_out");
    mid_out->mutable_data<float>(ctx.GetPlace());

    int n = ctx.Attr<int>("n");
    float alpha = ctx.Attr<float>("alpha");
    float beta = ctx.Attr<float>("beta");
    float k = ctx.Attr<float>("k");

    PADDLE_ENFORCE(n > 0, "n should >= 0");
    PADDLE_ENFORCE(alpha >= 0.0, "alpha should >= 0.0");
    PADDLE_ENFORCE(beta >= 0.0, "beta should >= 0.0");
    PADDLE_ENFORCE(k >= 0.0, "k should >= 0.0");

    printf("3\n");
    auto mid_out_v = framework::EigenVector<T>::Flatten(*mid_out);
    printf("4\n");
    mid_out_v.setConstant(k);

    printf("5\n");
    auto x_v = framework::EigenVector<T>::Flatten(*x);
    printf("6\n");
    const T* data = x_v.data();
    std::cout << data << std::endl;

    const int start = -(n - 1) / 2;
    const int end = start + n;

    /*
    for (int m = 0; m < N; m++) {
      const T* sample = data + m * one_sample_size;

      for (int i = 0; i < C; i++) {
        auto mid_data =
            mid_out_v.data() + m * one_sample_size + i * one_img_size;
        framework::EigenTensor<float, 1>::Type s =
            framework::EigenTensor<float, 1>::From(
                mid_data, framework::make_ddim({one_img_size}));

        for (int c = start; c <= end; c++) {
          int ch = i + c;
          if (ch >= 0 && ch < C) {
            auto input = framework::EigenTensor<T, 1>::From(
                (sample + ch * one_img_size),
                framework::make_ddim({one_img_size}));
            s += (input.square() * alpha);
          }
        }
      }
    }
    */

    auto e_mid = framework::EigenTensor<T, 4>::From(*mid_out);
    printf("e_mid 1\n");
    e_mid.setConstant(k);
    printf("e_mid 2\n");

    auto e_x = framework::EigenTensor<T, 4>::From(*x);
    for (int m = 0; m < N; m++) {
      for (int i = 0; i < C; i++) {
        for (int c = start; c <= end; c++) {
          int ch = i + c;
          if (ch >= 0 && ch < C) {
            printf("m:%d i:%d ch:%d\n", m, i, ch);
            auto s = e_mid.slice(Eigen::array<int, 4>({{m, i, 0, 0}}),
                                 Eigen::array<int, 4>({{1, 1, H, W}}));

            auto r = e_x.slice(Eigen::array<int, 4>({{m, ch, 0, 0}}),
                               Eigen::array<int, 4>({{1, 1, H, W}}));

            s += alpha * r.square();
          }
        }
      }
    }

    printf("out_e 3\n");
    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(ctx.GetEigenDevice<Place>()) =
        x_v * e_mid.reshape(Eigen::DSizes<int, 1>(e_mid.size())).pow(-beta);

    e_mid.device(ctx.GetEigenDevice<Place>()) = e_mid;
  }
};

/**
 * \brief Backward calculation for normalization with across maps.
 *
 * Function implementation:
 *
 * The implementation of this Function is derived from the
 * CrossMapNormalFunc implementation.
 *
 * InputGrad = OutputGrad * denoms ^ (-beta)
 *    -- upper
 *  + > (OutputGrad * OutputValue * (-2 * alpha * beta) / mid_out) * InputValue
 *    -- lower
 *
 * The data of inputs/outputs format is the same as the forward interface
 * and is NCHW.
 *
 * The upper and lower is the same as forward. The logic of the sum
 * is also the same as forward.
 */
template <typename Place, typename T>
class LRNGradKernel : public framework::OpKernel {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
