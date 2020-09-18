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

#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/segment_ops/segment_pooling.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSegmentSumCustomKernel(const Index input_outer_dim_size,
                                             const Index inner_dim_size,
                                             const Index output_outer_dim_size,
                                             const Index* segment_ids,
                                             const T* input, T* output,
                                             const Index total_stripe_count) {
  CUDA_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id) {
          platform::CudaAtomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += input[(input_outer_dim_index_base + j) * inner_dim_size +
                   segment_offset];
      // sum += __ldg(input + (input_outer_dim_index_base + j) * inner_dim_size
      // +segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    platform::CudaAtomicAdd(output + output_index, sum);
  }
}

template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSegmentMeanCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* segment_ids, const T* input,
    T* output, T* summed_ids, const Index total_stripe_count) {
  CUDA_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);

    if (segment_offset == 0) {
      T sum = T(0);
      for (Index j = 0; j < actual_stripe_height; j++) {
        Index current_output_segment_id =
            segment_ids[input_outer_dim_index_base + j];
        if (current_output_segment_id > last_output_segment_id) {
          if (last_output_segment_id == first_segment_id) {
            platform::CudaAtomicAdd(summed_ids + last_output_segment_id, sum);
          } else {
            *(summed_ids + last_output_segment_id) = sum;
          }
          sum = T(0);
        }
        sum += 1;
        last_output_segment_id = current_output_segment_id;
      }
      platform::CudaAtomicAdd(summed_ids + last_output_segment_id, sum);
    }
    __syncthreads();
    T sum = T(0);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        if (last_output_segment_id == first_segment_id) {
          platform::CudaAtomicAdd(output + output_index,
                                  sum / *(summed_ids + last_output_segment_id));
        } else {
          *(output + output_index) =
              sum / *(summed_ids + last_output_segment_id);
        }
        sum = T(0);
      }
      sum += input[(input_outer_dim_index_base + j) * inner_dim_size +
                   segment_offset];
      last_output_segment_id = current_output_segment_id;
    }
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    platform::CudaAtomicAdd(output + output_index,
                            sum / *(summed_ids + last_output_segment_id));
  }
}

template <typename T, typename Index, int OuterDimTileSize, typename Pool>
__global__ void SortedSegmentMinMaxCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* segment_ids, const T* input,
    T* output, const Index total_stripe_count, Pool pool) {
  CUDA_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T minmax = pool.initial();
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    // -1 is for the start value when interval_id = 0
    Index previous_segment_id = -1;
    if (input_outer_dim_index_base > 0) {
      previous_segment_id = segment_ids[input_outer_dim_index_base - 1];
    }
    for (Index interval_id = previous_segment_id + 1;
         interval_id < first_segment_id; ++interval_id) {
      *(output + interval_id * inner_dim_size + segment_offset) = 0;
    }

    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];

      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        if (last_output_segment_id == first_segment_id) {
          pool.atomic(output + output_index, minmax);
        } else {
          *(output + output_index) = minmax;
        }
        // reset the interval value which do not have corresponding ids.
        for (Index interval_index = 1;
             interval_index <
             current_output_segment_id - last_output_segment_id;
             ++interval_index) {
          *(output + output_index + interval_index * inner_dim_size) = 0;
        }
        minmax = pool.initial();
      }
      pool.compute(input[(input_outer_dim_index_base + j) * inner_dim_size +
                         segment_offset],
                   &minmax);
      last_output_segment_id = current_output_segment_id;
    }
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    pool.atomic(output + output_index, minmax);
  }
}

template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSegmentIndexGradKernel(const Index input_outer_dim_size,
                                             const Index inner_dim_size,
                                             const Index output_outer_dim_size,
                                             const Index* segment_ids,
                                             const T* input, const T* output,
                                             const T* out_grad, T* in_grad,
                                             const Index total_stripe_count) {
  CUDA_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      Index input_index =
          (input_outer_dim_index_base + j) * inner_dim_size + segment_offset;
      Index output_index =
          current_output_segment_id * inner_dim_size + segment_offset;
      if (input[input_index] == output[output_index]) {
        in_grad[input_index] = out_grad[output_index];
      }
    }
  }
}

template <class T>
class MaxPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(-FLT_MAX); }
  DEVICE inline void compute(const T& x, T* y) { *y = *y > x ? *y : x; }
  DEVICE inline T atomic(T* address, const T val) {
    return platform::CudaAtomicMax(address, val);
  }
};

template <class T>
class MinPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(FLT_MAX); }
  DEVICE inline void compute(const T& x, T* y) { *y = *y < x ? *y : x; }
  DEVICE inline T atomic(T* address, const T val) {
    return platform::CudaAtomicMin(address, val);
  }
};

template <class T>
class DataArrangeHelper {
 public:
  const T input_total_size;
  const T input_outer_dim_size;
  const T output_rows;
  T input_inner_dim_size;
  T total_stripe_count;
  const T OuterDimTileSize = 8;

  DataArrangeHelper(T a, T b, T c)
      : input_total_size(a), input_outer_dim_size(b), output_rows(c) {
    T input_outer_dim_num_stripe =
        (input_outer_dim_size + OuterDimTileSize - 1) / OuterDimTileSize;
    input_inner_dim_size = input_total_size / input_outer_dim_size;
    total_stripe_count = input_inner_dim_size * input_outer_dim_num_stripe;
  }
};

template <typename T, typename Index>
void SegmentPoolCUDAFunctor(const platform::CUDADeviceContext& ctx,
                            const framework::Tensor& input,
                            const framework::Tensor& segment_ids,
                            framework::Tensor* output,
                            framework::Tensor* summed_ids,
                            const std::string pooltype = "SUM") {
  const Index OuterDimTileSize = 8;
  auto h = DataArrangeHelper<Index>(input.numel(), segment_ids.dims()[0],
                                    output->dims()[0]);
  auto config = platform::GetGpuLaunchConfig1D(ctx, h.total_stripe_count);

  if (pooltype == "MEAN") {
    SortedSegmentMeanCustomKernel<T, Index, OuterDimTileSize><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
        h.input_outer_dim_size, h.input_inner_dim_size, h.output_rows,
        segment_ids.data<Index>(), input.data<T>(), output->data<T>(),
        summed_ids->data<T>(), h.total_stripe_count);
  } else if (pooltype == "SUM") {
    SortedSegmentSumCustomKernel<T, Index, OuterDimTileSize><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
        h.input_outer_dim_size, h.input_inner_dim_size, h.output_rows,
        segment_ids.data<Index>(), input.data<T>(), output->data<T>(),
        h.total_stripe_count);
  } else if (pooltype == "MAX") {
    MaxPool<T> pool;
    SortedSegmentMinMaxCustomKernel<T, Index, OuterDimTileSize, MaxPool<T>><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
        h.input_outer_dim_size, h.input_inner_dim_size, h.output_rows,
        segment_ids.data<Index>(), input.data<T>(), output->data<T>(),
        h.total_stripe_count, pool);
  } else if (pooltype == "MIN") {
    MinPool<T> pool;
    SortedSegmentMinMaxCustomKernel<T, Index, OuterDimTileSize, MinPool<T>><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
        h.input_outer_dim_size, h.input_inner_dim_size, h.output_rows,
        segment_ids.data<Index>(), input.data<T>(), output->data<T>(),
        h.total_stripe_count, pool);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported segment pooling operation, Only MEAN, SUM, MAX, MIN "
        "available, but got %s.",
        pooltype));
  }
}

template <typename T, typename Index>
void SegmentPoolCUDAGradFunctor(const platform::CUDADeviceContext& ctx,
                                const framework::Tensor& input,
                                const framework::Tensor& segment_ids,
                                const framework::Tensor& output,
                                const framework::Tensor& out_grad,
                                framework::Tensor* in_grad,
                                const std::string pooltype = "SUM") {
  const Index OuterDimTileSize = 8;
  auto h = DataArrangeHelper<Index>(input.numel(), segment_ids.dims()[0],
                                    output.dims()[0]);
  auto config = platform::GetGpuLaunchConfig1D(ctx, h.total_stripe_count);
  if (pooltype == "MAX" || pooltype == "MIN") {
    SortedSegmentIndexGradKernel<T, Index, OuterDimTileSize><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
        h.input_outer_dim_size, h.input_inner_dim_size, h.output_rows,
        segment_ids.data<Index>(), input.data<T>(), output.data<T>(),
        out_grad.data<T>(), in_grad->data<T>(), h.total_stripe_count);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported segment pooling grad operation, Only MAX, MIN "
        "available, but got %s.",
        pooltype));
  }
}

template <typename T>
__global__ void SimpleDiv(T* x, const T* y, const int len, const int dim) {
  for (int i = blockIdx.x; i < len; i += gridDim.x) {
    __shared__ T y_i;
    auto base = i * dim;
    if (threadIdx.x == 0) {
      y_i = y[i];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
      x[base + j] /= y_i;
    }
  }
}

template <typename T, typename IndexT>
class SegmentPoolFunctor<platform::CUDADeviceContext, T, IndexT> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& segments, framework::Tensor* output,
                  framework::Tensor* summed_ids = nullptr,
                  const std::string pooltype = "SUM") {
    SegmentPoolCUDAFunctor<T, IndexT>(context, input, segments, output,
                                      summed_ids, pooltype);
  }
};

template <typename T, typename IndexT>
class SegmentPoolGradFunctor<platform::CUDADeviceContext, T, IndexT> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& segments, framework::Tensor* in_grad,
                  const framework::Tensor* summed_ids = nullptr,
                  const std::string pooltype = "SUM") {
    if (pooltype == "MAX" || pooltype == "MIN") {
      SegmentPoolCUDAGradFunctor<T, IndexT>(context, input, segments, output,
                                            out_grad, in_grad, pooltype);
    } else if (pooltype == "MEAN") {
      framework::Tensor mean_grad;
      mean_grad.mutable_data<T>(input.dims(), context.GetPlace());
      framework::TensorCopy(out_grad, context.GetPlace(), context, &mean_grad);
      int len = output.dims()[0];
      int dim = output.numel() / len;
      auto config = platform::GetGpuLaunchConfig1D(context, len);
      SimpleDiv<T><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                     context.stream()>>>(mean_grad.data<T>(),
                                         summed_ids->data<T>(), len, dim);
      GPUGather<T, IndexT>(context, mean_grad, segments, in_grad);
    } else if (pooltype == "SUM") {
      GPUGather<T, IndexT>(context, out_grad, segments, in_grad);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported segment pooling operation, Only MEAN, SUM, MAX, MIN "
          "available, but got %s.",
          pooltype));
    }
  }
};

using CUDA = paddle::platform::CUDADeviceContext;
template class SegmentPoolFunctor<CUDA, float, int>;
template class SegmentPoolFunctor<CUDA, float, int64_t>;
template class SegmentPoolFunctor<CUDA, double, int>;
template class SegmentPoolFunctor<CUDA, double, int64_t>;
template class SegmentPoolGradFunctor<CUDA, float, int>;
template class SegmentPoolGradFunctor<CUDA, float, int64_t>;
template class SegmentPoolGradFunctor<CUDA, double, int>;
template class SegmentPoolGradFunctor<CUDA, double, int64_t>;

}  // namespace operators
}  // namespace paddle
