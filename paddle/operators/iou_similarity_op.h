/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
#include "paddle/platform/for_range.h"

template <typename T>
inline T IOUSimilarity(T xmin1, T ymin1, T xmax1, T ymax1, T xmin2, T ymin2,
                       T xmax2, T ymax2) {
  T area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
  T area2 = (ymax2 - ymin2) * (xmax2 - xmin2);
  T inter_xmax = std::min(xmax1, xmax2);
  T inter_ymax = std::min(ymax1, ymax2);
  T inter_xmin = std::max(xmin1, xmin2);
  T inter_ymin = std::max(ymin1, ymin2);
  T inter_height = std::max(inter_ymax - inter_ymin, static_cast<T>(0));
  T inter_width = std::max(inter_xmax - inter_xmin, static_cast<T>(0));
  T inter_area = inter_width * inter_height;
  T union_area = area1 + area2 - inter_area;
  T sim_score = inter_area / union_area;
  return sim_score;
}

template <typename T>
struct IOUSimilarityFunctor {
  IOUSimilarityFunctor(const T* x, const T* y, T* z, int cols)
      : x_(x), y_(y), z_(z), cols_(static_cast<size_t>(cols)) {}

  inline HOSTDEVICE void operator()(size_t row_id) const {
    T x_min1 = x_[row_id * 4];
    T y_min1 = x_[row_id * 4 + 1];
    T x_max1 = x_[row_id * 4 + 2];
    T y_max1 = x_[row_id * 4 + 3];
    for (int i = 0; i < cols_; ++i) {
      T x_min2 = y_[i * 4];
      T y_min2 = y_[i * 4 + 1];
      T x_max2 = y_[i * 4 + 2];
      T y_max2 = y_[i * 4 + 3];

      T sim = IOUSimilarity(x_min1, y_min1, x_max1, y_max1, x_min2, y_min2,
                            x_max2, y_max2);

      z_[row_id * cols_ + i] = sim;
    }
  }
  const T* x_;
  const T* y_;
  T* z_;
  const size_t cols_;
};

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class IOUSimilarityKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* in_x = ctx.Input<framework::Tensor>("X");
    const framework::Tensor* in_y = ctx.Input<framework::Tensor>("Y");
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");

    int x_n = in_x->dims()[0];
    int y_n = in_y->dims()[0];
    IOUSimilarityFunctor<T> functor(in_x->data<T>(), in_y->data<T>(),
                                    out->mutable_data<T>(ctx.GetPlace()), y_n);

    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(ctx.device_context()), x_n);
    for_range(functor);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
