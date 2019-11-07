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

#include "paddle/fluid/operators/math/prelu.h"

namespace paddle {
namespace operators {
namespace math {

static const int CUDA_NUM_THREADS = 1024;
static const int CUDA_MAX_NUM_BLOCKS = 65535;
inline static int GET_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void PReluChannelWiseKernel(const T *input, const T *alpha,
                                       T *output, int channel_num,
                                       size_t plane_size, size_t spatial_size,
                                       bool use_plane_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T *in = input + offset;
  T *out = output + offset;
  T scale = alpha[blockIdx.x % channel_num];

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T x = in[i];
    if (use_plane_size) {
      T s = alpha[i / plane_size];
      out[i] = (x > 0) ? x : s * x;
    } else {
      out[i] = (x > 0) ? x : scale * x;
    }
  }
}

template <typename T>
__global__ void PReluElementWiseKernel(const T *input, const T *alpha,
                                       T *output, int batch_size,
                                       size_t plane_size, size_t spatial_size,
                                       bool use_plane_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T *in = input + offset;
  auto channel_index = blockIdx.x % batch_size;
  const T *scale = alpha + channel_index * spatial_size;

  if (use_plane_size) scale = alpha;

  T *out = output + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T x = in[i];
    out[i] = (x > 0) ? x : scale[i] * x;
  }
}

template <typename T>
__global__ void PReluScalarKernel(const T *input, const T *alpha, T *output,
                                  size_t plane_size, size_t spatial_size,
                                  bool use_plane_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T *in = input + offset;
  T scale = *alpha;
  T *out = output + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T x = in[i];
    out[i] = (x > 0) ? x : scale * x;
  }
}

template <typename T>
void PreluChannelWiseDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *input, const T *alpha, T *output,
    std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t plane_size = input_shape[2] * input_shape[3];
  size_t spatial_size = plane_size;
  bool use_plane_size = false;
  if (unroll > CUDA_MAX_NUM_BLOCKS) {
    unroll = input_shape[0];
    spatial_size = input_shape[1] * input_shape[2] * input_shape[3];
    use_plane_size = true;
  }
  size_t num_threads = CUDA_NUM_THREADS;
  if (spatial_size < CUDA_NUM_THREADS) num_threads = spatial_size;
  CHECK_LE(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluChannelWiseKernel<<<unroll, num_threads, 0, stream>>>(
      input, alpha, output, input_shape[1], plane_size, spatial_size,
      use_plane_size);
}

template <typename T>
void PreluElementWiseDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *input, const T *alpha, T *output,
    std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t plane_size = input_shape[2] * input_shape[3];
  size_t spatial_size = plane_size;
  bool use_plane_size = false;
  if (unroll > CUDA_MAX_NUM_BLOCKS) {
    unroll = input_shape[0];
    spatial_size = input_shape[1] * input_shape[2] * input_shape[3];
    use_plane_size = true;
  }
  size_t num_threads = CUDA_NUM_THREADS;
  if (spatial_size < CUDA_NUM_THREADS) num_threads = spatial_size;
  CHECK_LE(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluElementWiseKernel<<<unroll, num_threads, 0, stream>>>(
      input, alpha, output, input_shape[0], plane_size, spatial_size,
      use_plane_size);
}

template <typename T>
void PreluScalarDirectCUDAFunctor<T>::operator()(cudaStream_t stream,
                                                 const T *input, const T *alpha,
                                                 T *output,
                                                 std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t plane_size = input_shape[2] * input_shape[3];
  size_t spatial_size = plane_size;
  bool use_plane_size = false;
  if (unroll > CUDA_MAX_NUM_BLOCKS) {
    unroll = input_shape[0];
    spatial_size = input_shape[1] * input_shape[2] * input_shape[3];
    use_plane_size = true;
  }
  size_t num_threads = CUDA_NUM_THREADS;
  if (spatial_size < CUDA_NUM_THREADS) num_threads = spatial_size;
  CHECK_LE(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluScalarKernel<<<unroll, num_threads, 0, stream>>>(
      input, alpha, output, plane_size, spatial_size, use_plane_size);
}

template class PreluChannelWiseDirectCUDAFunctor<float>;
template class PreluChannelWiseDirectCUDAFunctor<double>;

template class PreluElementWiseDirectCUDAFunctor<float>;
template class PreluElementWiseDirectCUDAFunctor<double>;

template class PreluScalarDirectCUDAFunctor<float>;
template class PreluScalarDirectCUDAFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
