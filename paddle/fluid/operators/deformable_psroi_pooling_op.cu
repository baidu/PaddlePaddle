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
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__device__ T bilinear_interp(const T *data,
                             const T x,
                             const T y,
                             const int width,
                             const int height) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  T dist_x = (T)(x - x1);
  T dist_y = (T)(y - y1);
  T value11 = data[y1 * width + x1];
  T value12 = data[y2 * width + x1];
  T value21 = data[y1 * width + x2];
  T value22 = data[y2 * width + x2];
  T value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) *
            dist_y * value12 + dist_x * (1 - dist_y) * value21 +
            dist_x * dist_y * value22;
  return value;
}

template <typename T>
__global__ void DeformablePSROIPoolForwardKernel(
    const int count, const T* bottom_data, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const T *bottom_rois,
    const T *bottom_trans, const int no_trans, const T trans_std,
    const int sample_per_part, const int output_dim, const int group_size,
    const int part_size, const int num_classes, const int channels_each_class,
    T* top_data, T* top_count, int* roi_batch_id_data) {
  CUDA_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
    int num_box = count / pooled_height / pooled_width /output_dim;
    const T *offset_bottom_rois = bottom_rois + n * 4;
    int roi_batch_ind = roi_batch_id_data[n];
    T roi_start_w = (T)(round(offset_bottom_rois[0])) * spatial_scale - 0.5;
    T roi_start_h = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_end_w = (T)(round(offset_bottom_rois[2]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
    T roi_height = max(roi_end_h - roi_start_h, 0.1);
    T bin_size_h = roi_height / (T)(pooled_height);
    T bin_size_w = roi_width / (T)(pooled_width);
    T sub_bin_size_h = bin_size_h / (T)(sample_per_part);
    T sub_bin_size_w = bin_size_w / (T)(sample_per_part);
    int part_h = floor((T)(ph) / pooled_height * part_size);
    int part_w = floor((T)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x = no_trans ? (T)(0) : bottom_trans[(((n * num_classes +
                class_id) * 2) * part_size + part_h) * part_size +
                part_w] * (T)trans_std;
    T trans_y = no_trans ? (T)(0) : bottom_trans[(((n * num_classes +
                class_id) * 2 + 1) * part_size + part_h) * part_size +
                part_w] * (T)trans_std;
    T wstart = (T)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = (T)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    T sum = 0;
    int count = 0;
    int gw = floor((T)(pw)*group_size / pooled_width);
    int gh = floor((T)(ph)*group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    const T* offset_bottom_data = bottom_data + (roi_batch_ind
                                  * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        T val = bilinear_interp(offset_bottom_data + c * height
                * width, w, h, width, height);
        sum += val;
        count++;
      }
    }
    top_data[index] = count == 0 ? (T)(0) : sum / count;
    top_count[index] = count;
  }
}


template <typename DeviceContext, typename T>
class DeformablePSROIPoolCUDAKernel : public framework::OpKernel<T>{
 public:
  void Compute(const framework::ExecutionContext& ctx)  const override{
    const Tensor* input = ctx.Input<Tensor>("Input");
    const LoDTensor* bbox = ctx.Input<LoDTensor>("ROIs");
    const Tensor* trans = ctx.Input<Tensor>("Trans");
    Tensor* out = ctx.Output<Tensor>("Output");
    out->mutable_data<T>(ctx.GetPlace());
    Tensor* top_count = ctx.Output<Tensor>("TopCount");
    top_count->mutable_data<T>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(top_count->dims(), out->dims(),
        "number of bbox should be same with number of output");
    const int no_trans = ctx.Attr<int>("no_trans");
    const float spatial_scale = ctx.Attr<float>("spatial_scale");
    const int output_dim = ctx.Attr<int>("output_dim");
    const int group_size = ctx.Attr<int>("group_size");
    const int pooled_size = ctx.Attr<int>("pooled_size");
    const int part_size = ctx.Attr<int>("part_size");
    const int sample_per_part = ctx.Attr<int>("sample_per_part");
    const float trans_std = ctx.Attr<float>("trans_std");

    const int batch = static_cast<int>(input->dims()[0]);
    const int channels = static_cast<int>(input->dims()[1]);
    const int height = static_cast<int>(input->dims()[2]);
    const int width = static_cast<int>(input->dims()[3]);
    const int channels_trans = no_trans ? 2 : trans->dims()[1];

    const int num_bbox = bbox->dims()[0];
    PADDLE_ENFORCE_EQ(num_bbox, out->dims()[0],
        "number of bbox should be same with number of output");
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int count = num_bbox * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class = no_trans ? output_dim
                                    : output_dim / num_classes;
    const T* bottom_data = input->data<T>();
    const T* bottom_rois = bbox->data<T>();
    const T* bottom_trans = no_trans ? NULL : trans->data<T>();
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({num_bbox});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto rois_lod = bbox->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(rois_batch_size, batch,
        "The rois_batch_size and imgs batch_size must be the same.");
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(num_bbox, rois_num_with_lod,
        "The rois_num from input and lod must be the same.");
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    auto& dev_ctx = ctx.cuda_device_context();
    auto& allocator = platform::DeviceTemporaryAllocator::
                      Instance().Get(dev_ctx);
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = allocator.Allocate(bytes);
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                 dev_ctx.stream());
    T *top_data = out->mutable_data<T>(ctx.GetPlace());
    T *top_count_data = top_count->mutable_data<T>(ctx.GetPlace());
    DeformablePSROIPoolForwardKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS,
        0,dev_ctx.stream()>>>(count, bottom_data, (T)spatial_scale, channels,
        height, width, pooled_height, pooled_width, bottom_rois, bottom_trans,
        no_trans, (T)trans_std, sample_per_part, output_dim, group_size,
        part_size, num_classes, channels_each_class, top_data, top_count_data,
        roi_id_data);
  }
};

template <typename T>
__global__ void DeformablePSROIPoolBackwardAccKernel(
    const int count, const T* top_diff, const T* top_count,
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int output_dim, T* bottom_data_diff,
    T* bottom_trans_diff, const T *bottom_data, const T *bottom_rois,
    const T *bottom_trans, const int no_trans, const T trans_std,
    const int sample_per_part, const int group_size, const int part_size,
    const int num_classes, const int channels_each_class,
    int* roi_batch_id_data) {
  CUDA_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
    int num_box = count / pooled_height / pooled_width /output_dim;
    const T *offset_bottom_rois = bottom_rois + n * 4;
    int roi_batch_ind = roi_batch_id_data[n];
    T roi_start_w = (T)(round(offset_bottom_rois[0])) * spatial_scale - 0.5;
    T roi_start_h = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_end_w = (T)(round(offset_bottom_rois[2]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_width = max(roi_end_w - roi_start_w, 0.1);
    T roi_height = max(roi_end_h - roi_start_h, 0.1);
    T bin_size_h = roi_height / (T)(pooled_height);
    T bin_size_w = roi_width / (T)(pooled_width);
    T sub_bin_size_h = bin_size_h / (T)(sample_per_part);
    T sub_bin_size_w = bin_size_w / (T)(sample_per_part);
    int part_h = floor((T)(ph) / pooled_height * part_size);
    int part_w = floor((T)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x = no_trans ? (T)(0) : bottom_trans[(((n * num_classes +
                class_id) * 2) * part_size + part_h) * part_size +
                part_w] * (T)trans_std;
    T trans_y = no_trans ? (T)(0) : bottom_trans[(((n * num_classes +
                class_id) * 2 + 1) * part_size + part_h) * part_size +
                part_w] * (T)trans_std;
    T wstart = (T)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = (T)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    if (top_count[index] <= 0) {
      continue;
    }
    T diff_val = top_diff[index] / top_count[index];
    const T* offset_bottom_data = bottom_data + roi_batch_ind
                                  * channels * height * width;
    T* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind
                                 * channels * height * width;
    int gw = floor((T)(pw)*group_size / pooled_width);
    int gh = floor((T)(ph)*group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        int x0 = floor(w);
        int x1 = ceil(w);
        int y0 = floor(h);
        int y1 = ceil(h);
        T dist_x = w - x0, dist_y = h - y0;
        T q00 = (1 - dist_x) * (1 - dist_y);
        T q01 = (1 - dist_x) * dist_y;
        T q10 = dist_x * (1 - dist_y);
        T q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;
        platform::CudaAtomicAdd(offset_bottom_data_diff + bottom_index_base
                                + y0 * width + x0, q00 * diff_val);
        platform::CudaAtomicAdd(offset_bottom_data_diff + bottom_index_base
                                + y1 * width + x0, q01 * diff_val);
        platform::CudaAtomicAdd(offset_bottom_data_diff + bottom_index_base
                                + y0 * width + x1, q10 * diff_val);
        platform::CudaAtomicAdd(offset_bottom_data_diff + bottom_index_base
                                + y1 * width + x1, q11 * diff_val);
        if (no_trans) {
          continue;
        }
        T U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
        T U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
        T U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
        T U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
        T diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y -
                    U00 * (1 - dist_y)) * trans_std * diff_val;
        diff_x *= roi_width;
        T diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x -
                    U00 * (1 - dist_x)) * trans_std * diff_val;
        diff_y *= roi_height;
        platform::CudaAtomicAdd(bottom_trans_diff + (((n * num_classes +
                                class_id) * 2) * part_size + part_h) *
                                part_size + part_w, diff_x);
        platform::CudaAtomicAdd(bottom_trans_diff + (((n * num_classes +
                                class_id) * 2 + 1) * part_size + part_h)
                                * part_size + part_w, diff_y);
      }
    }
  }
}

template <typename DeviceContext, typename T>
class DeformablePSROIPoolGradCUDAKernel : public framework::OpKernel<T>{
 public:
  void Compute(const framework::ExecutionContext& ctx)  const override{
    const Tensor* input = ctx.Input<Tensor>("Input");
    const LoDTensor* bbox = ctx.Input<LoDTensor>("ROIs");
    const Tensor* trans = ctx.Input<Tensor>("Trans");
    const Tensor* top_count = ctx.Input<Tensor>("TopCount");
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor *input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor *trans_grad = ctx.Output<Tensor>(framework::GradVarName("Trans"));
    input_grad->mutable_data<T>(ctx.GetPlace());
    trans_grad->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = ctx.cuda_device_context();
    set_zero(dev_ctx, input_grad, static_cast<T>(0));
    set_zero(dev_ctx, trans_grad, static_cast<T>(0));
    const int no_trans = ctx.Attr<int>("no_trans");
    const float spatial_scale = ctx.Attr<float>("spatial_scale");
    const int output_dim = ctx.Attr<int>("output_dim");
    const int group_size = ctx.Attr<int>("group_size");
    const int pooled_size = ctx.Attr<int>("pooled_size");
    const int part_size = ctx.Attr<int>("part_size");
    const int sample_per_part = ctx.Attr<int>("sample_per_part");
    const float trans_std = ctx.Attr<float>("trans_std");
    const int batch = static_cast<int>(input->dims()[0]);
    const int channels = static_cast<int>(input->dims()[1]);
    const int height = static_cast<int>(input->dims()[2]);
    const int width = static_cast<int>(input->dims()[3]);
    const int channels_trans = no_trans ? 2 : trans->dims()[1];
    const int num_bbox = bbox->dims()[0];
    const int pooled_height = pooled_size;
    const int pooled_width = pooled_size;
    const int count = num_bbox * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class = no_trans
                                    ? output_dim : output_dim / num_classes;
    const T* top_diff = output_grad->data<T>();
    const T* bottom_data = input->data<T>();
    const T* bottom_rois = bbox->data<T>();
    const T* bottom_trans = no_trans ? NULL : trans->data<T>();
    T* bottom_data_diff = input_grad->mutable_data<T>(ctx.GetPlace());
    T* bottom_trans_diff = no_trans ? NULL :
                           trans_grad->mutable_data<T>(ctx.GetPlace());
    const T* top_count_data = top_count->data<T>();
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({num_bbox});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto rois_lod = bbox->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(rois_batch_size, batch,
        "The rois_batch_size and imgs batch_size must be the same.");
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(num_bbox, rois_num_with_lod,
        "The rois_num from input and lod must be the same.");
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    auto& allocator = platform::DeviceTemporaryAllocator::
                      Instance().Get(dev_ctx);
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = allocator.Allocate(bytes);
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                 dev_ctx.stream());
    DeformablePSROIPoolBackwardAccKernel<<<GET_BLOCKS(count),
        CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
        count, top_diff, top_count_data, num_bbox, (T)spatial_scale,
        channels, height, width, pooled_height, pooled_width, output_dim,
        bottom_data_diff, bottom_trans_diff, bottom_data, bottom_rois,
        bottom_trans, no_trans, (T)trans_std, sample_per_part, group_size,
        part_size, num_classes, channels_each_class, roi_id_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(deformable_psroi_pooling,
    ops::DeformablePSROIPoolCUDAKernel<CUDA, float>,
    ops::DeformablePSROIPoolCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(deformable_psroi_pooling_grad,
    ops::DeformablePSROIPoolGradCUDAKernel<CUDA, float>,
    ops::DeformablePSROIPoolGradCUDAKernel<CUDA, double>);
