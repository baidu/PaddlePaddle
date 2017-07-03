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

#include "paddle/platform/device_context.h"
#include "gtest/gtest.h"


TEST(DeviceContext, CudaDevice) {
  int count = paddle::platform::GetDeviceCount();
  for (int i = 0; i < count; i++) {
    paddle::platform::CudaDeviceContext* device_context = new paddle::platform::CudaDeviceContext(i);
    __attribute__((unused)) Eigen::GpuDevice gpu_device = device_context->eigen_device();
    /* TODO dynamic load cudnn, cublas and curand libraries
    __attribute__((unused)) cudnnHandle_t cudnn_handle = device_context->cudnn_handle();
    __attribute__((unused)) cublasHandle_t cublas_handle = device_context->cublas_handle();
    __attribute__((unused)) curandGenerator_t curand_handle = device_context->curand_generator();
    */
    delete device_context;
  }
}
