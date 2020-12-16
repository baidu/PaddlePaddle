// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/service/heter_server.h"
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/fleet/heter_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace distributed {

std::shared_ptr<HeterServer> HeterServer::s_instance_ = NULL;

void HeterServer::RegisterServiceHandler(std::string message_name,
                                         HeterServiceHandler func) {
  service_.RegisterServiceHandler(message_name, func);
}

void HeterServer::StartHeterService() {
  std::unique_lock<std::mutex> running_lock(mutex_);
  server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  if (server_.Start(endpoint_.c_str(), &options) != 0) {
    VLOG(0) << "heter server start fail";
  } else {
    VLOG(0) << "heter server start success! listen on " << endpoint_;
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_ready_);
    ready_ = 1;
  }
  condition_ready_.notify_all();

  cv_.wait(running_lock, [&] {
    VLOG(0) << "Heter Server Stop.";
    return stoped_;
  });
}

void HeterServer::SetEndPoint(std::string& endpoint) {
  endpoint_ = endpoint;
  service_.SetEndpoint(endpoint);
}

void HeterServer::SetFanin(int& fan_in) { service_.SetFanin(fan_in); }

void HeterServer::WaitServerReady() {
  std::unique_lock<std::mutex> lock(this->mutex_ready_);
  condition_ready_.wait(lock, [=] { return this->ready_ == 1; });
}

int32_t HeterService::stop_profiler(const PsRequestMessage& request,
                                    PsResponseMessage& response,
                                    brpc::Controller* cntl) {
  platform::DisableProfiler(
      platform::EventSortingKey::kDefault,
      string::Sprintf("heter_worker_%s_profile", endpoint_));
  return 0;
}

int32_t HeterService::start_profiler(const PsRequestMessage& request,
                                     PsResponseMessage& response,
                                     brpc::Controller* cntl) {
  platform::EnableProfiler(platform::ProfilerState::kAll);
  return 0;
}

int32_t HeterService::stop_heter_worker(const PsRequestMessage& request,
                                        PsResponseMessage& response,
                                        brpc::Controller* cntl) {
  auto client_id = request.client_id();
  stop_cpu_worker_set_.insert(client_id);
  if (stop_cpu_worker_set_.size() == fan_in_) {
    is_exit_ = true;
  }
  return 0;
}

}  // end namespace distributed
}  // end namespace paddle
