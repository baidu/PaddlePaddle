// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/details/var_utils.h"

#include <algorithm>

#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace framework {
namespace details {

template <typename T>
__global__ void CheckNanInfKernel(const T* value, const size_t numel,
                                  int print_value) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  T sum = static_cast<T>(0.0);
  // Todo(wangxi). simd speed up
  for (size_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
    sum += (value[i] - value[i]);
  }

  if (isnan(sum) || isinf(sum)) {
  } else {
    return;
  }

  if (print_value) {
    for (size_t i = tid; i < numel; i += blockDim.x) {
      if (isnan(value[i]) || isinf(value[i])) {
        printf("idx:%u value:%f\n", i, value[i]);
        // use param control whether print more value
        if (i < numel) {
          printf("idx:%u value:%f\n", i + 1, value[i + 1]);
        }
      }
    }
  }
  __syncthreads();
  // abort or not
  if (true) {
    PADDLE_ENFORCE(0, "===ERROR: find nan or inf===");
  }
}

template <typename T>
void CheckNanInf(const T* value, const size_t numel, int print_value) {
  T sum = static_cast<T>(0.0);
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    printf("===ERROR: has nan or inf===");
    if (print_value) {
      for (int i = 0; i < numel; ++i) {
        printf("idx:%u value:%f\n", i, value[i]);
      }
    }
    PADDLE_ENFORCE_EQ(1, 0, "===ERROR: find nan or inf===");
  }
}

template <typename DeviceContext>
struct CheckNanInfTool {
  template <typename T>
  void run(const std::string& var_name, const framework::Tensor& tensor,
           const platform::Place& place, int print_value,
           typename std::enable_if<std::is_integral<T>::value>::type* = 0);

  template <typename T>
  void run(
      const std::string& var_name, const framework::Tensor& tensor,
      const platform::Place& place, int print_value,
      typename std::enable_if<std::is_floating_point<T>::value>::type* = 0);
};

template <typename DeviceContext>
template <typename T>
void CheckNanInfTool<DeviceContext>::run(
    const std::string& var_name, const framework::Tensor& tensor,
    const platform::Place& place, int print_value,
    typename std::enable_if<std::is_integral<T>::value>::type*) {
  VLOG(10) << var_name << " need not to check, it's type is not float point";
}

template <>
template <typename T>
void CheckNanInfTool<platform::CUDADeviceContext>::run(
    const std::string& var_name, const framework::Tensor& tensor,
    const platform::Place& place, int print_value,
    typename std::enable_if<std::is_floating_point<T>::value>::type*) {
  auto* dev_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
      platform::DeviceContextPool::Instance().Get(tensor.place()));
  const size_t threads = 1024;
  size_t blocks = std::min(128ul, (tensor.numel() + threads - 1) / threads);
  CheckNanInfKernel<<<blocks, threads, 0, dev_ctx->stream()>>>(
      tensor.data<T>(), tensor.numel(), 1);
}

template <>
template <typename T>
void CheckNanInfTool<platform::CPUDeviceContext>::run(
    const std::string& var_name, const framework::Tensor& tensor,
    const platform::Place& place, int print_value,
    typename std::enable_if<std::is_floating_point<T>::value>::type*) {
  platform::DeviceContextPool::Instance().Get(tensor.place());
}

struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& var_name,
                       const framework::Tensor& tensor,
                       const platform::Place& place)
      : var_name_(var_name), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() const {
    int is_print_value = 1;

    if (platform::is_gpu_place(tensor_.place())) {
#ifdef PADDLE_WITH_CUDA
      CheckNanInfTool<platform::CUDADeviceContext> tools;
      tools.run<T>(var_name_, tensor_, place_, is_print_value);
#else
      PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
      return;
    }

    CheckNanInfTool<platform::CPUDeviceContext> tools;
    tools.run<T>(var_name_, tensor_, place_, is_print_value);
  }

  std::string var_name_;
  const framework::Tensor& tensor_;
  const platform::Place& place_;
};

void EnforceNoNanOrInf(const std::string& op_type,
                       const framework::Scope& scope,
                       const std::string& var_name,
                       const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var, "can't find var:%s", var_name);

  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<framework::SelectedRows>()) {
    tensor = &var->Get<framework::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  TensorCheckerVisitor vistor(var_name, *tensor, place);
  VisitDataType(tensor->type(), vistor);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
