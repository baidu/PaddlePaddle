#include <iostream>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;


template <typename T>
T bilinear_interp(const T *data, const T x,const T y,
                  const int width, const int height)
{
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
  T value = (1 - dist_x) * (1 - dist_y) * value11 + (1 - dist_x) * \
            dist_y * value12 + dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
  return value;
}


template <typename T>
void DeformablePSROIPoolForwardCPUKernel(const int count, const T* bottom_data,
                                         const T spatial_scale, const int channels,
                                         const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         const T* bottom_rois, const T* bottom_trans,
                                         const int no_trans, const float trans_std,
                                         const int sample_per_part, const int output_dim,
                                         const int group_size, const int part_size,
                                         const int num_classes, const int channels_each_class,
                                         T* top_data, T* top_count, const int batch_size, 
                                         int* roi_batch_id_data, const LoDTensor* rois)

{
  for (int ix = 0; ix < count; ix++)
  {  
    int pw = ix % pooled_width;    
    int ph = (ix / pooled_width) % pooled_height;
    int ctop = (ix / pooled_width / pooled_height) % output_dim;
    int n = ix / pooled_width / pooled_height / output_dim;
    int num_box = count / pooled_height / pooled_width /output_dim;
    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(num_box, rois_num_with_lod,
                      "The rois_num from input and lod must be the same.");

    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 4;
    int roi_batch_ind = roi_batch_id_data[n];
    T roi_start_w = (T)(round(offset_bottom_rois[0])) * spatial_scale - 0.5;
    T roi_start_h = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_end_w = (T)(round(offset_bottom_rois[2]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
  
    // Force too small ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, T(0.1)); //avoid 0
    T roi_height = std::max(roi_end_h - roi_start_h, T(0.1));
    
    T bin_size_h = roi_height / (T)(pooled_height);
    T bin_size_w = roi_width / (T)(pooled_width);

    T sub_bin_size_h = bin_size_h / (T)(sample_per_part);
    T sub_bin_size_w = bin_size_w / (T)(sample_per_part);

    int part_h = floor((T)(ph) / pooled_height * part_size);
    int part_w = floor((T)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x = no_trans ? (T)(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * (T)trans_std;
    T trans_y = no_trans ? (T)(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * (T)trans_std;
    
    T wstart = (T)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = (T)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    T sum = 0;
    int count_time = 0;
    int gw = floor((T)(pw)*group_size / pooled_width);
    int gh = floor((T)(ph)*group_size / pooled_height);
    gw = std::min(std::max(gw, 0), group_size - 1);
    gh = std::min(std::max(gh, 0), group_size - 1);
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = std::min(std::max(w, T(0.)), T(width - 1.));
        h = std::min(std::max(h, T(0.)), height - T(1.));
        int c = (ctop * group_size + gh) * group_size + gw;
        T val = bilinear_interp(offset_bottom_data + c * height * width, w, h, width, height);
        sum += val;
        count_time++;
      }
    }
    top_data[ix] = count_time == 0 ? (T)(0) : sum / count_time;
    top_count[ix] = count_time;
    }
}


template <typename DeviceContext, typename T>
class DeformablePSROIPoolCPUKernel : public framework::OpKernel<T>
{
  public:
    void Compute(const framework::ExecutionContext& ctx) const override{
      auto* input = ctx.Input<Tensor>("Input");
      auto* rois = ctx.Input<LoDTensor>("ROIs");
      auto* trans = ctx.Input<Tensor>("Trans");
      auto* out = ctx.Output<Tensor>("Output");
      out->mutable_data<T>(ctx.GetPlace());
      auto* top_count = ctx.Output<Tensor>("Top_count");
      top_count->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<DeviceContext, T> set_zero;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      set_zero(dev_ctx, out, static_cast<T>(0));
      set_zero(dev_ctx, top_count, static_cast<T>(0));
      const int num_rois = rois->dims()[0];
      PADDLE_ENFORCE_EQ(num_rois, out->dims()[0], 
                        "number of rois should be same with number of output");
      PADDLE_ENFORCE_EQ(top_count->dims(), out->dims(), 
                        "number of top_count should be same with number of output"); 
      framework::Tensor roi_batch_id_list;
      roi_batch_id_list.Resize({num_rois});
      int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(ctx.GetPlace());        
      auto no_trans = ctx.Attr<int>("no_trans");
      auto spatial_scale = ctx.Attr<float>("spatial_scale");
      auto output_dim = ctx.Attr<int>("output_dim");
      auto group_size = ctx.Attr<int>("group_size");
      auto pooled_size = ctx.Attr<int>("pooled_size");
      auto part_size = ctx.Attr<int>("part_size");
      auto sample_per_part = ctx.Attr<int>("sample_per_part");
      auto trans_std = ctx.Attr<float>("trans_std");
      int batch = static_cast<int>(input->dims()[0]);
      int channels = static_cast<int>(input->dims()[1]);
      int height = static_cast<int>(input->dims()[2]);
      int width = static_cast<int>(input->dims()[3]);
      int channels_trans = no_trans ? 2 : trans->dims()[1];
      
      auto pooled_height = pooled_size;
      auto pooled_width = pooled_size;
      auto count = num_rois * output_dim * pooled_height * pooled_width;
      auto num_classes = no_trans ? 1 : channels_trans / 2;
      auto channels_each_class = no_trans ? output_dim : output_dim / num_classes;

      const T* bottom_data = input->data<T>();
      const T* bottom_rois = rois->data<T>();
      const T* bottom_trans = no_trans ? NULL : trans->data<T>();
        
      T* top_data = out->mutable_data<T>(ctx.GetPlace());
      T* top_count_data = top_count->mutable_data<T>(ctx.GetPlace());
      DeformablePSROIPoolForwardCPUKernel(
        count, bottom_data, (T)spatial_scale, channels, height, width, 
        pooled_height, pooled_width, bottom_rois, bottom_trans, no_trans, 
        trans_std, sample_per_part, output_dim, group_size, part_size, num_classes, 
        channels_each_class, top_data, top_count_data, batch, roi_batch_id_data, rois);
      }
};
    
template <typename T>
void DeformablePSROIPoolBackwardAccCPUKernel(const int count, const T* top_diff, const T* top_count, 
                                             const int num_rois,const T spatial_scale, const int channels,
                                             const int height, const int width, const int pooled_height, 
                                             const int pooled_width, const int output_dim,
                                             T* bottom_data_diff, T* bottom_trans_diff,
                                             const T* bottom_data, const T* bottom_rois,
                                             const T* bottom_trans, const int no_trans,
                                             const float trans_std, const int sample_per_part,
                                             const int group_size, const int part_size,
                                             const int num_classes, const int channels_each_class,
                                             const int batch_size, int* roi_batch_id_data, 
                                             const LoDTensor* rois)
{
  for (int index = 0; index < count; index++)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
      
    int num_box = count / pooled_height / pooled_width /output_dim;
    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(rois_batch_size, batch_size,
                      "The rois_batch_size and imgs batch_size must be the same.");
      
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(num_box, rois_num_with_lod,
                      "The rois_num from input and lod must be the same.");

    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
      
    // [start, end) interval for spatial sampling
    const T* offset_bottom_rois = bottom_rois + n * 4;
    int roi_batch_ind = roi_batch_id_data[n];
    T roi_start_w = (T)(round(offset_bottom_rois[0])) * spatial_scale - 0.5;
    T roi_start_h = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_end_w = (T)(round(offset_bottom_rois[2]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;

    // Force too small ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, T(0.1)); //avoid 0
    T roi_height = std::max(roi_end_h - roi_start_h, T(0.1));

    T bin_size_h = roi_height / (T)(pooled_height);
    T bin_size_w = roi_width / (T)(pooled_width);

    T sub_bin_size_h = bin_size_h / (T)(sample_per_part);
    T sub_bin_size_w = bin_size_w / (T)(sample_per_part);

    int part_h = floor((T)(ph) / pooled_height * part_size);
    int part_w = floor((T)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x = no_trans ? (T)(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * (T)trans_std;
    T trans_y = no_trans ? (T)(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * (T)trans_std;
    T wstart = (T)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = (T)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (top_count[index] <= 0)
    {
      continue;
    }
    T diff_val = top_diff[index] / top_count[index];
    const T* offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
    T* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
    int gw = floor((T)(pw)*group_size / pooled_width);
    int gh = floor((T)(ph)*group_size / pooled_height);
    gw = std::min(std::max(gw, 0), group_size - 1);
    gh = std::min(std::max(gh, 0), group_size - 1);
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
        {
          T w = wstart + iw * sub_bin_size_w;
          T h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
          {
            continue;
          }
          w = std::min(std::max(w, T(0.)), T(width - 1.));
          h = std::min(std::max(h, T(0.)), T(height - 1.));
          int c = (ctop * group_size + gh) * group_size + gw;
          // backward on feature
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
          T* offset_bottom_data_diff_addr00 =  offset_bottom_data_diff + bottom_index_base + y0 * width + x0;
          T* offset_bottom_data_diff_addr01 =  offset_bottom_data_diff + bottom_index_base + y1 * width + x0;
          T* offset_bottom_data_diff_addr10 =  offset_bottom_data_diff + bottom_index_base + y0 * width + x1;
          T* offset_bottom_data_diff_addr11 =  offset_bottom_data_diff + bottom_index_base + y1 * width + x1;

          *offset_bottom_data_diff_addr00 = *offset_bottom_data_diff_addr00 + q00 * diff_val;
          *offset_bottom_data_diff_addr01 = *offset_bottom_data_diff_addr01 + q01 * diff_val;
          *offset_bottom_data_diff_addr10 = *offset_bottom_data_diff_addr10 + q10 * diff_val;
          *offset_bottom_data_diff_addr11 = *offset_bottom_data_diff_addr11 + q11 * diff_val;

          if (no_trans)
          {
            continue;
          }
          T U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
          T U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
          T U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
          T U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
          T diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y)) * trans_std * diff_val;
          diff_x *= roi_width;
          T diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x)) * trans_std * diff_val;
          diff_y *= roi_height;
        
          T* offset_bottom_trans_diff_x =  bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w;
          T* offset_bottom_trans_diff_y = bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w;

          *offset_bottom_trans_diff_x =  *offset_bottom_trans_diff_x + diff_x;
          *offset_bottom_trans_diff_y =  *offset_bottom_trans_diff_y + diff_y;
        }
    }
  }
}

template <typename DeviceContext, typename T>
class DeformablePSROIPoolGradCPUKernel : public framework::OpKernel<T>{
  public:  
    void Compute(const framework::ExecutionContext& ctx)  const override{
      auto* input = ctx.Input<Tensor>("Input");
      auto* rois = ctx.Input<LoDTensor>("ROIs");
      auto* trans = ctx.Input<Tensor>("Trans");
      auto* top_count = ctx.Input<Tensor>("Top_count");
      auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
      auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
      input_grad->mutable_data<T>(ctx.GetPlace());
      auto* trans_grad = ctx.Output<Tensor>(framework::GradVarName("Trans"));
      trans_grad->mutable_data<T>(ctx.GetPlace());
      
      math::SetConstant<DeviceContext, T> set_zero;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      set_zero(dev_ctx, input_grad, static_cast<T>(.0));
      set_zero(dev_ctx, trans_grad, static_cast<T>(.0));
      auto no_trans = ctx.Attr<int>("no_trans");
      auto spatial_scale = ctx.Attr<float>("spatial_scale");
      auto output_dim = ctx.Attr<int>("output_dim");
      auto group_size = ctx.Attr<int>("group_size");
      auto pooled_size = ctx.Attr<int>("pooled_size");
      auto part_size = ctx.Attr<int>("part_size");
      auto sample_per_part = ctx.Attr<int>("sample_per_part");
      auto trans_std = ctx.Attr<float>("trans_std");
      
      const int batch = static_cast<int>(input->dims()[0]);
      const int channels = static_cast<int>(input->dims()[1]);
      const int height = static_cast<int>(input->dims()[2]);
      const int width = static_cast<int>(input->dims()[3]);
      const int channels_trans = no_trans ? 2 : trans->dims()[1];
      const int num_rois = rois->dims()[0];
      const int pooled_height = pooled_size;
      const int pooled_width = pooled_size;
      const int count = num_rois * output_dim * pooled_height * pooled_width;
      const int num_classes = no_trans ? 1 : channels_trans / 2;
      const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
      
      Tensor roi_batch_id_list;
      roi_batch_id_list.Resize({num_rois});
      int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(ctx.GetPlace());
      const T* top_diff = output_grad->data<T>();
      const T* bottom_data = input->data<T>();
      const T* bottom_rois = rois->data<T>();
      const T* bottom_trans = no_trans ? NULL : trans->data<T>();
      T* bottom_data_diff = input_grad->mutable_data<T>(ctx.GetPlace());
      T* bottom_trans_diff = no_trans ? NULL : trans_grad->mutable_data<T>(ctx.GetPlace());
      const T* top_count_data = top_count->data<T>();

      DeformablePSROIPoolBackwardAccCPUKernel(
        count, top_diff, top_count_data, num_rois, (T)spatial_scale, channels, height, width,
        pooled_height, pooled_width, output_dim, bottom_data_diff, bottom_trans_diff,
        bottom_data, bottom_rois, bottom_trans, no_trans, (T)trans_std, sample_per_part,
        group_size, part_size, num_classes, channels_each_class, batch, roi_batch_id_data, rois);
    }
};

} //namespace operators
} //namespace paddle
