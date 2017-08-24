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
#include "paddle/platform/device_context.h"

#define kNUM_STREAMS 16

namespace paddle {
namespace platform {

class DeviceContextManager {
 public:
  DeviceContextManager();
  ~DeviceContextManager();

  DeviceContext& GetDeviceContext(Place& place);
  DeviceContext& GetIODeviceContext(Place& place);

 private:
  CPUDeviceContext* cpu_context_{nullptr};
#ifndef PADDLE_ONLY_CPU
  std::vector<std::vector<CUDADeviceContext*>> cuda_contexts_;
  std::vector<CUDADeviceContext*> cuda_io_contexts_;
  std::vector<int> gpu_cnt_;
  int device_count_;
#endif
};
}  // namespace platform
}  // namespace paddle
