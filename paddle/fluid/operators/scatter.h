/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <cstring>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/place.h"
#include "unordered_set"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

/**
  * Return the updated array pointer, use blas or eigen lib to optimize time
 * cost
 */
template <typename T, typename IndexT = int>
typename std::enable_if<std::is_floating_point<T>::value>::type
elementwise_inner_add(const framework::ExecutionContext& ctx,
                      const T* src_pointer, const T* dist_pointer,
                      T* result_dist_pointer, const framework::Tensor& src,
                      framework::Tensor* dist, const int& src_indice,
                      const IndexT& dist_indice, const int& slice_size,
                      const size_t& slice_bytes) {
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);

  blas.VADD(slice_size, src_pointer + src_indice * slice_size,
            dist_pointer + dist_indice * slice_size,
            result_dist_pointer + dist_indice * slice_size);
}

template <typename T, typename IndexT = int>
typename std::enable_if<!std::is_floating_point<T>::value>::type
elementwise_inner_add(const framework::ExecutionContext& ctx,
                      const T* src_pointer, const T* dist_pointer,
                      T* result_dist_pointer, const framework::Tensor& src,
                      framework::Tensor* dist, const int& src_indice,
                      const IndexT& dist_indice, const int& slice_size,
                      const size_t& slice_bytes) {
  auto src_slice = src.Slice(src_indice, src_indice + 1);
  auto dist_slice = dist->Slice(dist_indice, dist_indice + 1);

  auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
  auto eigen_dist = framework::EigenVector<T>::Flatten(dist_slice);

  eigen_dist += eigen_src;
}
/**
 * Return an updated tensor from source tensor, scattered according to indice:
 * dst[i] = src[indice[i]]
 * input[src]: type-T source Tensor
 * input[indice]: type-IndexT indice Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void ScatterAssign(const platform::DeviceContext& ctx, const Tensor& src,
                   const Tensor& indice, Tensor* output) {
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true);
  // check indice of shape 1-D
  if (indice.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(indice.dims()[1], 1,
                      "indice.dims()[1] shold be 1 when indice.dims().size() == "
                      "2 in scatter_op.");
  } else {
    PADDLE_ENFORCE_EQ(indice.dims().size(), 1,
                      "indice.dims().size() shold be 1 or 2 in scatter_op.");
  }
  int indice_size = indice.dims()[0];

  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_indice = indice.data<IndexT>();
  T* p_output = output->data<T>();

  // check src shape and dst shape shold match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE_EQ(src_dims[i], dst_dims[i]);

  // slice size
  size_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int i = 0; i < indice_size; ++i) {
    IndexT indice_ = p_indice[i];
    memcpy(p_output + indice_ * slice_size, p_src + i * slice_size, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void ScatterAssignAdd(const framework::ExecutionContext& ctx, const Tensor& src,
                      const Tensor& indice, Tensor* output) {
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.device_context().GetPlace()),
                    true);
  // check indice of shape 1-D
  PADDLE_ENFORCE(indice.dims().size() == 1 ||
                     (indice.dims().size() == 2 && indice.dims()[1] == 1),
                 "");
  int indice_size = indice.dims()[0];

  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_indice = indice.data<IndexT>();

  const T* p_output = output->data<T>();
  T* result_p_output = output->data<T>();

  // check src shape and dst shape shold match
  for (int i = 1; i < src_dims.size(); i++)
    PADDLE_ENFORCE_EQ(src_dims[i], dst_dims[i]);

  // slice size
  size_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t& slice_bytes = slice_size * sizeof(T);

  // if not in overwrite mode, need to init output data
  for (int i = 0; i < indice_size; ++i) {
    const IndexT& indice_ = p_indice[i];
    memset(result_p_output + slice_size * indice_, 0, slice_bytes);
  }

  // if not in overwrite mode, need to init output data
  for (int i = 0; i < indice_size; ++i) {
    const IndexT& indice_ = p_indice[i];
    elementwise_inner_add<T, IndexT>(ctx, p_src, p_output, result_p_output, src,
                                     output, i, indice_, slice_size,
                                     slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void ScatterNdAdd(const framework::ExecutionContext& ctx, const Tensor& update,
                  const Tensor& indice, Tensor* output) {
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.device_context().GetPlace()),
                    true, "It shold be running on the CPU");

  // update.shape = indice.shape[:-1] + output.shape[indice.shape[-1]:]
  auto indice_dims = indice.dims();
  auto indice_dims_size = indice_dims.size();

  auto output_dims = output->dims();
  auto output_dims_size = output_dims.size();

  const T* p_update = update.data<T>();
  const IndexT* p_indice = indice.data<IndexT>();
  T* result_p_output = output->data<T>();
  const T* p_output = output->data<T>();

  // final dim
  int64_t end_size = indice_dims[indice_dims_size - 1];
  // remain dim
  auto remain_ddim = framework::slice_ddim(indice_dims, 0, indice_dims_size - 1);
  int64_t remain_numel = framework::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < output_dims_size; ++i) {
    slice_size *= output_dims[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < remain_numel; ++i) {
    IndexT indice_ = 0;
    IndexT temp = 1;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      IndexT indice_value = p_indice[i * end_size + j];
      indice_ += (indice_value * temp);
      temp *= output_dims[j];
    }
    elementwise_inner_add<T, IndexT>(ctx, p_update, p_output, result_p_output,
                                     update, output, i, indice_, slice_size,
                                     slice_bytes);
  }
}

}  // namespace operators
}  // namespace paddle
